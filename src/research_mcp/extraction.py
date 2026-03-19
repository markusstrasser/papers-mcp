"""Structured extraction tables — Elicit-style column-based extraction across papers."""

import asyncio
import json
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

EXTRACT_MODEL = "gemini-3-flash-preview"
MAX_CONCURRENT = 5
MAX_TEXT_CHARS = 30_000  # Truncate full text to this if no RCS summaries

COLUMN_PRESETS = {
    "clinical": [
        {"name": "sample_size", "prompt": "Total sample size (N)"},
        {"name": "study_design", "prompt": "Study design (RCT, cohort, case-control, etc.)"},
        {"name": "population", "prompt": "Study population description"},
        {"name": "main_finding", "prompt": "Primary finding or conclusion"},
        {"name": "effect_size", "prompt": "Effect size with confidence interval if reported"},
    ],
}


async def _extract_one(
    client: genai.Client,
    paper: dict,
    columns: list[dict],
    semaphore: asyncio.Semaphore,
) -> dict:
    """Extract columns from a single paper."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "Untitled")

    text = (paper.get("full_text") or "")[:MAX_TEXT_CHARS]
    if not text.strip():
        return {"paper_id": paper_id, "title": title, "error": "no text available"}

    col_spec = "\n".join(f'- "{c["name"]}": {c["prompt"]}' for c in columns)

    prompt = (
        f"Extract the following fields from this paper.\n\n"
        f"Paper: {title}\n\n"
        f"Text:\n{text}\n\n"
        f"Fields to extract:\n{col_spec}\n\n"
        f'Return JSON with these fields. Use "N/A" if not found. Be precise — quote numbers exactly.'
    )

    async with semaphore:
        response = await client.aio.models.generate_content(
            model=EXTRACT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

    raw = response.text or "{}"
    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"error": "failed to parse extraction output"}

    return {"paper_id": paper_id, "title": title, **extracted}


async def extract_table_async(
    papers: list[dict],
    columns: list[dict],
) -> list[dict]:
    """Extract structured columns from multiple papers concurrently."""
    client = genai.Client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _extract_one(client, paper, columns, semaphore)
        for paper in papers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rows = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("Extraction failed: %s", r)
            continue
        rows.append(r)

    return rows
