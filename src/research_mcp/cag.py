"""Cache-Augmented Generation — stuff full papers into Gemini's 1M context."""

import logging
import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _is_retryable_genai(exc: BaseException) -> bool:
    """Retry transient Gemini failures: 5xx (model overloaded / 503) and 429
    (rate limit). Mirrors discovery.py's S2 retry — the single synthesis call
    otherwise hard-fails ask_papers on a transient overload."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    return isinstance(exc, genai_errors.ClientError) and getattr(exc, "code", None) == 429


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception(_is_retryable_genai),
)
def _generate_with_retry(client: genai.Client, model: str, contents: str, config):
    return client.models.generate_content(model=model, contents=contents, config=config)

# Tiered models: cheap classification tier for broad many-paper sweeps,
# capable synthesis tier for focused analysis + the RCS answer path.
# (Both previously defaulted to gemini-3-flash-preview, silently collapsing the
# tiering — the "focused" answer used the cheap classification model. 3.5-flash
# is the synthesis default per operator routing; 2026-06-01.)
MODEL_BROAD = os.environ.get("CAG_MODEL_BROAD", "gemini-3-flash-preview")
MODEL_FOCUSED = os.environ.get("CAG_MODEL_FOCUSED", "gemini-3.5-flash")

# ~4 chars per token on average for English text
CHARS_PER_TOKEN = 4
MAX_INPUT_TOKENS = 1_000_000
# Reserve tokens for the question + system prompt + output
RESERVED_TOKENS = 70_000
MAX_CORPUS_TOKENS = MAX_INPUT_TOKENS - RESERVED_TOKENS
MAX_CORPUS_CHARS = MAX_CORPUS_TOKENS * CHARS_PER_TOKEN

SYSTEM_PROMPT = """\
You are a research assistant with access to the full text of scientific papers.
Answer the question using ONLY the provided papers. For every claim, cite the
source paper by [Author Year] or [Title]. If the papers don't contain enough
information, say so explicitly — do not hallucinate.

Be precise with numbers, methods, and conclusions. Distinguish between what
the paper claims, what its data shows, and what remains uncertain."""


def ask_corpus(
    question: str,
    papers: list[dict],
    model: str | None = None,
) -> dict:
    """Answer a question using full paper texts stuffed into Gemini context.

    Args:
        question: Research question.
        papers: List of dicts with 'title', 'authors', 'year', 'full_text'.
        model: Override model selection. If None, auto-selects based on corpus size.

    Returns:
        dict with 'answer', 'model', 'tokens_used', 'papers_included'.
    """
    if not papers:
        return {"answer": "No papers with full text in corpus. Use fetch_paper to download PDFs first.", "model": None, "tokens_used": 0, "papers_included": 0}

    # Build corpus text, fitting as many papers as possible
    corpus_parts = []
    total_chars = 0
    included = 0

    for p in papers:
        text = p.get("full_text", "")
        if not text:
            continue
        header = f"=== {p.get('title', 'Untitled')} ({', '.join((p.get('authors') or [])[:3])}, {p.get('year', '?')}) ===\n"
        entry = header + text + "\n\n"
        if total_chars + len(entry) > MAX_CORPUS_CHARS:
            break
        corpus_parts.append(entry)
        total_chars += len(entry)
        included += 1

    if not corpus_parts:
        return {"answer": "Papers have no extractable text.", "model": None, "tokens_used": 0, "papers_included": 0}

    # Auto-select model: broad (many papers, cheap) vs focused (few papers, capable)
    if model is None:
        model = MODEL_BROAD if included > 30 else MODEL_FOCUSED

    corpus = "".join(corpus_parts)
    est_tokens = len(corpus) // CHARS_PER_TOKEN + len(question) // CHARS_PER_TOKEN

    logger.info(f"CAG: {included} papers, ~{est_tokens:,} tokens, model={model}")

    client = genai.Client()
    response = _generate_with_retry(
        client,
        model,
        f"Papers:\n\n{corpus}\n\nQuestion: {question}",
        types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    answer = response.text or "(empty response)"
    usage = response.usage_metadata
    tokens_used = usage.total_token_count if usage else est_tokens

    return {
        "answer": answer,
        "model": model,
        "tokens_used": tokens_used,
        "papers_included": included,
    }


def ask_corpus_rcs(
    question: str,
    evidence: list[dict],
    model: str | None = None,
) -> dict:
    """Answer a question using pre-scored evidence summaries (RCS path).

    Unlike ask_corpus which stuffs full papers, this uses only the scored
    and summarized chunks from prepare_evidence. More focused, less noise.

    Args:
        question: Research question.
        evidence: List of dicts from prepare_evidence (summary, relevance, paper_title).
        model: Override model selection. If None, uses focused model.

    Returns:
        dict with 'answer', 'model', 'tokens_used', 'evidence_used'.
    """
    if not evidence:
        return {"answer": "No relevant evidence found.", "model": None, "tokens_used": 0, "evidence_used": 0}

    # Build context from scored summaries
    parts = []
    total_chars = 0
    included = 0

    for e in evidence:
        entry = f"=== {e['paper_title']} (relevance: {e['relevance']}/10) ===\n{e['summary']}\n\n"
        if total_chars + len(entry) > MAX_CORPUS_CHARS:
            break
        parts.append(entry)
        total_chars += len(entry)
        included += 1

    if model is None:
        model = MODEL_FOCUSED  # RCS already filtered — use focused model

    corpus = "".join(parts)
    est_tokens = len(corpus) // CHARS_PER_TOKEN + len(question) // CHARS_PER_TOKEN

    logger.info(f"CAG-RCS: {included} evidence chunks, ~{est_tokens:,} tokens, model={model}")

    client = genai.Client()
    response = _generate_with_retry(
        client,
        model,
        f"Evidence (sorted by relevance):\n\n{corpus}\n\nQuestion: {question}",
        types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=8192,
        ),
    )

    answer = response.text or "(empty response)"
    usage = response.usage_metadata
    tokens_used = usage.total_token_count if usage else est_tokens

    return {
        "answer": answer,
        "model": model,
        "tokens_used": tokens_used,
        "evidence_used": included,
    }
