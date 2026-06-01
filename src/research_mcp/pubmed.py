"""PubMed (NCBI E-utilities) backend — ELink cross-database linking only.

Biomedical *search* lives in the EuropePMC backend (one call returns full
abstracts; PubMed would need esearch→esummary→efetch and esummary doesn't even
return abstracts). The one capability EuropePMC and Semantic Scholar can't
match is NCBI ELink: traversing PubMed's curated links into other Entrez
databases — a PMID's associated genes, compounds, nucleotide/protein records,
or PMC full text. That's all this module exposes.

Etiquette: NCBI asks for ``tool`` and ``email`` on every request and grants
3 req/s anonymously, 10 req/s with an API key. Same sync min-interval gate as
the EuropePMC backend (safe inside FastMCP's worker-thread-run sync tools).
"""

import logging
import time

import httpx

log = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Entrez databases ELink can target from PubMed.
LINK_TARGETS = {"gene", "pccompound", "nuccore", "protein", "pubmed", "pmc"}


class PubMed:
    """NCBI ELink client with a sync min-interval rate gate."""

    def __init__(
        self,
        api_key: str | None = None,
        email: str | None = None,
        tool: str = "research-mcp",
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key
        self.email = email
        self.tool = tool
        qps = 10.0 if api_key else 3.0
        self._min_interval = 1.0 / qps
        self._last_request = 0.0
        self._own_client = client is None
        self.client = client or httpx.Client(base_url=EUTILS_BASE, timeout=30)

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _common_params(self) -> dict:
        params = {"tool": self.tool, "retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def elink(self, pmids: list[str], target_db: str = "gene") -> dict:
        """Map each PMID to linked IDs in ``target_db`` via NCBI ELink.

        One HTTP request handles the whole batch of PMIDs. Returns
        ``{"target_db": ..., "links": {pmid: [linked_ids]}}``.
        """
        if target_db not in LINK_TARGETS:
            raise ValueError(
                f"target_db must be one of {sorted(LINK_TARGETS)}, got {target_db!r}"
            )
        if not pmids:
            return {"target_db": target_db, "links": {}}

        params = self._common_params()
        params.update({"dbfrom": "pubmed", "db": target_db})
        # Repeat the id param per PMID so ELink returns one linkset per input
        # PMID (a single comma-joined id collapses them into one linkset).
        query = [("id", str(p)) for p in pmids]

        self._wait()
        try:
            resp = self.client.get(
                "/elink.fcgi", params=[*params.items(), *query]
            )
            self._last_request = time.monotonic()
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            raise RuntimeError(
                f"NCBI ELink {exc.response.status_code}: {body}"
            ) from exc

        return {"target_db": target_db, "links": self._parse_linksets(data)}

    @staticmethod
    def _parse_linksets(data: dict) -> dict[str, list[str]]:
        links: dict[str, list[str]] = {}
        for linkset in data.get("linksets", []):
            seed_ids = linkset.get("ids", [])
            seed = str(seed_ids[0]) if seed_ids else None
            if seed is None:
                continue
            # NCBI may return several linksetdbs (e.g. pubmed_gene +
            # pubmed_gene_rif) — flatten and dedupe, preserving first-seen order.
            seen: dict[str, None] = {}
            for linksetdb in linkset.get("linksetdbs", []):
                for x in linksetdb.get("links", []):
                    seen.setdefault(str(x), None)
            links[seed] = list(seen)
        return links

    def close(self) -> None:
        if self._own_client:
            self.client.close()
