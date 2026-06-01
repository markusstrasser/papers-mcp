"""EuropePMC backend — biomedical literature search + citation graph.

EuropePMC mirrors PubMed/MEDLINE + PMC + Agricola + preprints and, unlike the
NCBI E-utilities, returns the full abstract in a single ``resultType=core``
search call (no esearch→esummary→efetch dance). We use it for two things:

1. A non–Semantic-Scholar biomedical search backend for ``search_papers``.
2. A citation/reference fallback when S2 is rate-limited (403). EuropePMC only
   indexes biomedical literature, so the caller MUST gate this fallback on a
   biomedical identifier (PMID/PMCID) — see ``server.traverse_citations``.

Full-text (JATS XML) retrieval is deliberately NOT implemented here: feeding
JATS-derived text into the content-addressed paper store would produce a
different hash than the PDF path for the same paper, duplicating corpus records
and breaking attestation. That integration, if ever wanted, needs its own
canonicalization design.

Rate limiting is a simple sync min-interval gate. research-mcp is a single
long-lived process making serial agent-driven calls; the cross-process
file-lock limiter used by google-deepmind/science-skills solves a concurrency
problem we don't have here. If parallel fetch subagents are ever added, that
(fcntl.flock + Retry-After + X-Throttling-Control backpressure) is the upgrade
path.
"""

import logging
import time

import httpx

log = logging.getLogger(__name__)

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"


class EuropePMC:
    """EuropePMC REST client (no auth) with a sync min-interval rate gate."""

    def __init__(self, qps: float = 3.0, client: httpx.Client | None = None):
        self._min_interval = 1.0 / qps
        self._last_request = 0.0
        self._own_client = client is None
        self.client = client or httpx.Client(base_url=EPMC_BASE, timeout=30)

    def _wait(self) -> None:
        """Block until min-interval since the last request has elapsed.

        Sync sleep is safe: callers are sync FastMCP tools that FastMCP runs in
        a worker thread, so this never blocks the event loop.
        """
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _get(self, path: str, params: dict) -> dict:
        self._wait()
        try:
            resp = self.client.get(path, params=params)
            self._last_request = time.monotonic()
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            # Include the response body — gives the agent something to act on.
            body = exc.response.text[:500]
            raise RuntimeError(
                f"EuropePMC {exc.response.status_code} for {path}: {body}"
            ) from exc

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Search EuropePMC, returning records shaped like ``search_papers``.

        Uses ``resultType=core`` so each hit carries the full ``abstractText``
        in the same response (no follow-up fetch needed).
        """
        data = self._get(
            "/search",
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": min(limit, 100),
            },
        )
        results = data.get("resultList", {}).get("result", [])
        return [self._normalize(r) for r in results[:limit]]

    def citations(self, source: str, ext_id: str, limit: int = 100) -> list[dict]:
        """Papers that cite the given record (S2 ``get_citations`` analogue)."""
        return self._citation_graph("citations", source, ext_id, limit)

    def references(self, source: str, ext_id: str, limit: int = 100) -> list[dict]:
        """Papers referenced by the given record (S2 ``get_references``)."""
        return self._citation_graph("references", source, ext_id, limit)

    def _citation_graph(
        self, kind: str, source: str, ext_id: str, limit: int
    ) -> list[dict]:
        data = self._get(
            f"/{source}/{ext_id}/{kind}",
            {"format": "json", "pageSize": min(limit, 1000)},
        )
        # citations → {"citationList": {"citation": [...]}}
        # references → {"referenceList": {"reference": [...]}}
        if kind == "citations":
            items = data.get("citationList", {}).get("citation", [])
        else:
            items = data.get("referenceList", {}).get("reference", [])
        return [self._normalize_citation(c) for c in items[:limit]]

    @staticmethod
    def _normalize(raw: dict) -> dict:
        """Map a EuropePMC core record to the canonical paper dict shape."""
        source = raw.get("source", "")
        epmc_id = raw.get("id", "")
        pmid = raw.get("pmid")
        pmcid = raw.get("pmcid")
        doi = raw.get("doi")
        ext_ids: dict[str, str] = {}
        if pmid:
            ext_ids["PubMed"] = pmid
        if pmcid:
            ext_ids["PubMedCentral"] = pmcid
        if doi:
            ext_ids["DOI"] = doi
        author_string = raw.get("authorString", "") or ""
        authors = [a.strip() for a in author_string.split(",") if a.strip()]
        year = None
        if raw.get("pubYear"):
            try:
                year = int(raw["pubYear"])
            except (ValueError, TypeError):
                year = None
        return {
            "paper_id": f"EPMC:{source}:{epmc_id}" if epmc_id else (doi or pmid or ""),
            "doi": doi,
            "title": raw.get("title", ""),
            "abstract": raw.get("abstractText"),
            "authors": authors,
            "year": year,
            "venue": raw.get("journalTitle"),
            "citation_count": raw.get("citedByCount", 0),
            "external_ids": ext_ids,
            "open_access_url": None,
            "source_backend": "europepmc",
        }

    @staticmethod
    def _normalize_citation(raw: dict) -> dict:
        """Citation/reference list entries are lighter than core records."""
        pmid = raw.get("id") if raw.get("source") == "MED" else None
        return {
            "paper_id": f"EPMC:{raw.get('source', '')}:{raw.get('id', '')}",
            "doi": raw.get("doi"),
            "title": raw.get("title", ""),
            "year": raw.get("pubYear"),
            "citation_count": raw.get("citedByCount") or 0,
            "external_ids": {"PubMed": pmid} if pmid else {},
        }

    def close(self) -> None:
        if self._own_client:
            self.client.close()
