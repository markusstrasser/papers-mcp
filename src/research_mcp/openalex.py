"""OpenAlex API client — fallback for Semantic Scholar."""

import hashlib
import time
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from research_mcp.db import PaperDB

OA_BASE = "https://api.openalex.org"


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 (rate limit) and 5xx server errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


def _reconstruct_abstract(inv_index: dict | None) -> str | None:
    """Reconstruct plaintext abstract from OpenAlex inverted index format.

    OpenAlex stores abstracts as {"word": [pos1, pos2], ...}.
    """
    if not inv_index:
        return None
    words: dict[int, str] = {}
    for word, positions in inv_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[k] for k in sorted(words))


class OpenAlex:
    def __init__(self, db: PaperDB, api_key: str | None = None, email: str | None = None):
        self.db = db
        params = {}
        if api_key:
            params["api_key"] = api_key
        elif email:
            params["mailto"] = email
        self.client = httpx.Client(base_url=OA_BASE, params=params, timeout=30)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception(_is_retryable),
    )
    def search(self, query: str, limit: int = 10) -> list[dict]:
        cache_key = f"openalex:search:{hashlib.md5(f'{query}:{limit}'.encode()).hexdigest()}"
        cached = self.db.get_cache(cache_key)
        if cached is not None:
            return cached
        resp = self.client.get(
            "/works",
            params={"search": query, "per_page": limit},
        )
        if not resp.is_success:
            self._raise_with_backoff(resp)
        data = resp.json().get("results", [])
        results = [self._normalize(w) for w in data]
        self.db.set_cache(cache_key, results)
        return results

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(min=2, max=30),
        retry=retry_if_exception(_is_retryable),
    )
    def get_paper(self, paper_id: str) -> dict | None:
        """Get a paper by OpenAlex ID (W...) or DOI."""
        # DOIs need the URL prefix for OpenAlex lookup
        if paper_id.startswith("10."):
            lookup = f"https://doi.org/{paper_id}"
        else:
            lookup = paper_id

        cache_key = f"openalex:paper:{lookup}"
        cached = self.db.get_cache(cache_key)
        if cached is not None:
            return cached
        resp = self.client.get(f"/works/{lookup}")
        if resp.status_code == 404:
            return None
        if not resp.is_success:
            self._raise_with_backoff(resp)
        result = self._normalize(resp.json())
        self.db.set_cache(cache_key, result)
        return result

    def _raise_with_backoff(self, resp: httpx.Response) -> None:
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait_secs = int(retry_after) if retry_after and retry_after.isdigit() else 5
            time.sleep(min(wait_secs, 30))
        resp.raise_for_status()

    def _normalize(self, raw: dict) -> dict:
        # Extract OpenAlex ID (strip URL prefix)
        oa_id = raw.get("id", "")
        if oa_id.startswith("https://openalex.org/"):
            oa_id = oa_id[len("https://openalex.org/"):]

        # DOI — strip prefix
        doi = raw.get("doi") or ""
        if doi.startswith("https://doi.org/"):
            doi = doi[len("https://doi.org/"):]
        doi = doi or None

        # Authors
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in raw.get("authorships", [])
        ]

        # Venue
        primary_loc = raw.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        venue = source.get("display_name")

        # External IDs
        ids_raw = raw.get("ids") or {}
        external_ids = {}
        if doi:
            external_ids["DOI"] = doi
        if ids_raw.get("pmid"):
            pmid = ids_raw["pmid"]
            if pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
                pmid = pmid[len("https://pubmed.ncbi.nlm.nih.gov/"):]
            external_ids["PubMed"] = pmid
        if ids_raw.get("openalex"):
            external_ids["OpenAlex"] = oa_id

        # OA URL
        oa = raw.get("open_access") or {}
        open_access_url = oa.get("oa_url")

        return {
            "paper_id": oa_id,
            "doi": doi,
            "title": raw.get("title") or raw.get("display_name", ""),
            "abstract": _reconstruct_abstract(raw.get("abstract_inverted_index")),
            "authors": authors,
            "year": raw.get("publication_year"),
            "venue": venue,
            "citation_count": raw.get("cited_by_count", 0),
            "external_ids": external_ids,
            "open_access_url": open_access_url,
        }
