"""RCS (Relevance-scoring, Chunking, Summarization) for evidence preparation.

PaperQA2 ablation: removing score+summarize step drops accuracy (p<0.001).
Chunks paper text, scores each chunk against a query via Gemini Flash,
returns sorted summaries with relevance scores.
"""

import asyncio
import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

SCORE_MODEL = "gemini-3-flash-preview"
MAX_CONCURRENT = 5  # Gemini free tier safe


def chunk_text(text: str, max_chars: int = 3000, overlap: int = 300) -> list[str]:
    """Split text on paragraph boundaries, fallback to sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try paragraph boundary (double newline)
        boundary = text.rfind("\n\n", start + max_chars // 2, end)
        if boundary == -1:
            # Try sentence boundary
            for sep in (". ", ".\n", "? ", "! "):
                boundary = text.rfind(sep, start + max_chars // 2, end)
                if boundary != -1:
                    boundary += len(sep) - 1  # Include the punctuation
                    break
        if boundary == -1 or boundary <= start:
            boundary = end  # Hard cut

        chunks.append(text[start:boundary])
        start = max(boundary - overlap, start + 1)

    return chunks


async def _score_one(
    client: genai.Client,
    query: str,
    chunk: str,
    paper_title: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Score a single chunk for relevance to the query."""
    prompt = (
        f"Rate how relevant this text excerpt is to the research question.\n\n"
        f"Question: {query}\n\n"
        f"Paper: {paper_title}\n\n"
        f"Excerpt:\n{chunk}\n\n"
        f'Respond with JSON only: {{"summary": "2-3 sentence summary of key content", "relevance": N}}\n'
        f"where relevance is 0 (irrelevant) to 10 (directly answers the question)."
    )

    async with semaphore:
        response = await client.aio.models.generate_content(
            model=SCORE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )

    text = response.text or "{}"
    try:
        result = json.loads(text)
        return {
            "summary": result.get("summary", ""),
            "relevance": float(result.get("relevance", 0)),
        }
    except (json.JSONDecodeError, ValueError):
        return {"summary": chunk[:200], "relevance": 0.0}


async def score_chunks(
    client: genai.Client,
    query: str,
    chunks: list[str],
    paper_id: str,
    paper_title: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Score all chunks for a single paper concurrently."""
    tasks = [
        _score_one(client, query, chunk, paper_title, semaphore)
        for chunk in chunks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scored = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Chunk scoring failed for %s chunk %d: %s", paper_id, i, result)
            continue
        scored.append({
            "summary": result["summary"],
            "relevance": result["relevance"],
            "paper_id": paper_id,
            "paper_title": paper_title,
            "chunk_index": i,
        })

    return scored


async def prepare_evidence_async(
    query: str,
    papers: list[dict],
    min_score: float = 3.0,
) -> list[dict]:
    """Chunk, score, and sort evidence from multiple papers.

    Returns sorted list of evidence chunks with summaries and relevance scores.
    """
    client = genai.Client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    all_scored: list[dict] = []

    for paper in papers:
        text = paper.get("full_text", "")
        if not text:
            continue

        paper_id = paper.get("paper_id", "unknown")
        paper_title = paper.get("title", "Untitled")

        chunks = chunk_text(text)
        scored = await score_chunks(
            client, query, chunks, paper_id, paper_title, semaphore,
        )
        all_scored.extend(scored)

    # Filter by min_score and sort by relevance (descending)
    filtered = [s for s in all_scored if s["relevance"] >= min_score]
    filtered.sort(key=lambda s: s["relevance"], reverse=True)

    return filtered
