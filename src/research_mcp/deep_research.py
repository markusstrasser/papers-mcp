"""Gemini Deep Research — autonomous multi-step web research via Interactions API."""

import asyncio
import logging
import time

from google import genai

logger = logging.getLogger(__name__)

AGENT = "deep-research-pro-preview-12-2025"
POLL_INTERVAL = 10  # seconds between status checks
DEFAULT_TIMEOUT = 600  # 10 minutes


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

    # Extract text and citations from outputs
    report_parts = []
    citations = []
    thinking_parts = []

    for output in interaction.outputs or []:
        if output.type == "text":
            report_parts.append(output.text)
            for ann in output.annotations or []:
                url = getattr(ann, "url", None) or ann.source
                if url:
                    citations.append({
                        "url": url,
                        "start": ann.start_index,
                        "end": ann.end_index,
                    })
        elif output.type == "thought":
            thinking_parts.append(output.text if hasattr(output, "text") else str(output))

    report = "\n".join(report_parts)

    # Dedupe citations by URL
    seen = set()
    unique_citations = []
    for c in citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique_citations.append(c)

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

    if interaction.status == "completed" and interaction.outputs:
        report_parts = []
        citations = []
        for output in interaction.outputs:
            if output.type == "text":
                report_parts.append(output.text)
                for ann in output.annotations or []:
                    url = getattr(ann, "url", None) or ann.source
                    if url:
                        citations.append({"url": url})
        result["report"] = "\n".join(report_parts)
        result["report_chars"] = len(result["report"])
        seen = set()
        result["citations"] = [c for c in citations if c["url"] not in seen and not seen.add(c["url"])]

    return result
