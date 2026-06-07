"""Gemini Deep Research — autonomous multi-step web research via Interactions API."""

import asyncio
import logging
import time

from google import genai

logger = logging.getLogger(__name__)

AGENT = "deep-research-pro-preview-12-2025"
POLL_INTERVAL = 10  # seconds between status checks
DEFAULT_TIMEOUT = 600  # 10 minutes


def _extract_from_steps(steps) -> tuple[list[str], list[dict], list[str]]:
    """Pull report text, citations, and thinking summaries from an Interaction.

    google-genai 2.x restructured the response: the flat 1.x `interaction.outputs`
    (items with .type/.text/.annotations) became `interaction.steps` — a union
    discriminated by `.type`. Report text + citations now live inside
    ModelOutputStep.content[] → TextContent(.text, .annotations); thinking is
    ThoughtStep.summary. Defensive getattr so a schema tweak degrades, not crashes.
    """
    report_parts: list[str] = []
    citations: list[dict] = []
    thinking_parts: list[str] = []
    for step in steps or []:
        stype = getattr(step, "type", None)
        if stype == "model_output":
            for part in getattr(step, "content", None) or []:
                if getattr(part, "type", None) != "text":
                    continue
                report_parts.append(getattr(part, "text", "") or "")
                for ann in getattr(part, "annotations", None) or []:
                    url = (getattr(ann, "url", None) or getattr(ann, "uri", None)
                           or getattr(ann, "source", None))
                    if url:
                        citations.append({
                            "url": url,
                            "start": getattr(ann, "start_index", None),
                            "end": getattr(ann, "end_index", None),
                        })
        elif stype == "thought":
            summary = getattr(step, "summary", None)
            if summary:
                thinking_parts.append(summary if isinstance(summary, str) else str(summary))
    return report_parts, citations, thinking_parts


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for c in citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique


async def run_deep_research(
    query: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    thinking_summaries: bool = True,
) -> dict:
    """Run autonomous deep research and poll until completion.

    Args:
        query: Research question or topic.
        timeout: Max seconds to wait (default 600). Deep Research can take up to 60min.
        thinking_summaries: Include thinking summaries in output.

    Returns:
        dict with report text, citations, status, and timing.
    """
    client = genai.Client()
    t0 = time.monotonic()

    # Build config
    kwargs: dict = {
        "agent": AGENT,
        "input": query,
        "background": True,
        "store": True,
    }
    if thinking_summaries:
        kwargs["agent_config"] = {"type": "deep-research", "thinking_summaries": "auto"}

    logger.info("Deep Research: starting interaction for %r", query[:100])
    interaction = await client.aio.interactions.create(**kwargs)
    interaction_id = interaction.id
    logger.info("Deep Research: interaction %s created (status=%s)", interaction_id, interaction.status)

    # Poll until terminal state
    terminal = {"completed", "failed", "cancelled", "incomplete"}
    while interaction.status not in terminal:
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            # Try to cancel
            try:
                await client.aio.interactions.cancel(interaction_id)
            except Exception:
                pass
            return {
                "status": "timeout",
                "interaction_id": interaction_id,
                "elapsed_seconds": round(elapsed),
                "error": f"Timed out after {timeout}s. Use get_deep_research to check later.",
            }
        await asyncio.sleep(POLL_INTERVAL)
        interaction = await client.aio.interactions.get(interaction_id)
        logger.debug("Deep Research: %s status=%s (%.0fs)", interaction_id, interaction.status, elapsed)

    elapsed = round(time.monotonic() - t0)

    if interaction.status != "completed":
        return {
            "status": interaction.status,
            "interaction_id": interaction_id,
            "elapsed_seconds": elapsed,
            "error": f"Research ended with status: {interaction.status}",
        }

    # Extract text and citations from steps (google-genai 2.x response shape)
    report_parts, citations, thinking_parts = _extract_from_steps(interaction.steps)
    report = "\n".join(report_parts)
    unique_citations = _dedupe_citations(citations)

    # Usage stats
    usage = {}
    if interaction.usage:
        u = interaction.usage
        for field in ("input_tokens", "output_tokens", "total_tokens", "cached_tokens"):
            if hasattr(u, field):
                val = getattr(u, field)
                if val is not None:
                    usage[field] = val

    result = {
        "status": "completed",
        "interaction_id": interaction_id,
        "elapsed_seconds": elapsed,
        "report": report,
        "report_chars": len(report),
        "citations": unique_citations,
        "citation_count": len(unique_citations),
        "usage": usage,
    }

    if thinking_parts:
        result["thinking_summary"] = "\n".join(thinking_parts)

    return result


async def get_deep_research(interaction_id: str) -> dict:
    """Check status / retrieve results of a previously started deep research."""
    client = genai.Client()
    interaction = await client.aio.interactions.get(interaction_id)

    result = {
        "status": interaction.status,
        "interaction_id": interaction_id,
    }

    if interaction.status == "completed" and getattr(interaction, "steps", None):
        report_parts, citations, _ = _extract_from_steps(interaction.steps)
        result["report"] = "\n".join(report_parts)
        result["report_chars"] = len(result["report"])
        result["citations"] = _dedupe_citations(citations)

    return result
