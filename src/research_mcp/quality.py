"""Evidence quality assessment — mechanical checks, not composite grades.

Components:
  - Crossref retraction check
  - OpenAlex funding source lookup
  - Gemini Flash Lite feature extraction (study design, sample size, organism, candidate gene)
  - Hard vetoes: RETRACTED, CANDIDATE_GENE, NON_HUMAN_ONLY, CASE_REPORT_ONLY
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

# -- Constants ----------------------------------------------------------------

CANDIDATE_GENES_PROBLEMATIC = {
    "5-HTTLPR", "SLC6A4", "COMT", "Val158Met", "BDNF", "Val66Met",
    "DRD4", "DRD2", "MAOA", "MAOA-uVNTR", "FKBP5", "OXTR",
    "CHRNA5", "APOE",
}

INDUSTRY_KEYWORDS = {
    "pfizer", "novartis", "roche", "gsk", "merck", "abbott", "bayer",
    "sanofi", "amgen", "gilead", "herbalife", "now foods", "thorne",
    "nature's way", "garden of life", "nordic naturals", "metagenics",
    "nutricology", "jarrow", "solgar", "usana", "gnc",
}

GOV_KEYWORDS = {
    "nih", "nsf", "nhmrc", "erc", "wellcome", "mrc", "cihr", "dfg", "anr",
    "national institute", "national science",
}

_UA = "research-mcp/1.0; mailto:markus@strasser.me"
_TIMEOUT = httpx.Timeout(10.0)


# -- Dataclass ----------------------------------------------------------------

@dataclass
class PaperQuality:
    paper_id: str
    # Hard vetoes (binary, high-precision)
    retracted: bool = False
    retraction_detail: str = ""
    is_candidate_gene: bool = False
    organism: str = "unknown"
    is_case_report_only: bool = False
    # Informational context
    funding_source: str = "unknown"
    funding_entities: list[str] = field(default_factory=list)
    # Study characteristics (Flash Lite extracted)
    study_design: str = "unknown"
    sample_size: int | None = None
    is_meta_analysis: bool = False
    # Citation context (on-demand, not implemented yet)
    citation_stance: dict | None = None
    # Hard veto summary
    vetoed: bool = False
    veto_reasons: list[str] = field(default_factory=list)
    # Missingness tracking
    missing_fields: list[str] = field(default_factory=list)
    # Metadata
    assessed_at: str = ""
    checks_run: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> PaperQuality:
        return cls(**json.loads(raw))


# -- Mechanical checks --------------------------------------------------------

def check_retraction(doi: str, pmid: str | None = None) -> tuple[bool, str]:
    """Check Crossref + PubMed for retraction status. Returns (is_retracted, detail).

    Tries Crossref first (structured retraction metadata). Falls back to PubMed
    if Crossref doesn't have the DOI or misses the retraction.
    """
    # --- Crossref ---
    crossref_miss = False
    try:
        r = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as e:
        log.warning("Crossref request failed for %s: %s", doi, e)
        crossref_miss = True
        r = None

    if r and r.status_code == 200:
        try:
            data = r.json()["message"]
        except (KeyError, ValueError):
            data = {}

        # Check update-to with type "retraction"
        for update in data.get("update-to", []):
            if update.get("type") == "retraction":
                return True, f"Retraction via update-to: {update.get('label', 'retracted')}"

        # Check updated-by with type "retraction" or source "retraction-watch"
        for update in data.get("updated-by", []):
            if update.get("type") == "retraction" or update.get("source") == "retraction-watch":
                return True, f"Retraction via updated-by: {update.get('label', 'retracted')} (DOI: {update.get('DOI', '?')})"

        # Check relation.is-retracted-by
        retracted_by = data.get("relation", {}).get("is-retracted-by", [])
        if retracted_by:
            ids = [rb.get("id", "?") for rb in retracted_by]
            return True, f"Retracted by: {', '.join(ids)}"

        # Check if title starts with "RETRACTED:"
        for title in data.get("title", []):
            if isinstance(title, str) and title.upper().startswith("RETRACTED:"):
                return True, "Title marked as retracted"

    elif r and r.status_code == 404:
        crossref_miss = True

    # --- PubMed fallback (if Crossref missed or no retraction found) ---
    resolved_pmid = pmid
    if not resolved_pmid and not crossref_miss:
        # Try OpenAlex for PMID
        try:
            oa = httpx.get(
                f"https://api.openalex.org/works/doi:{doi}",
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
            )
            if oa.status_code == 200:
                ids = oa.json().get("ids", {})
                pmid_url = ids.get("pmid", "")
                if pmid_url:
                    resolved_pmid = pmid_url.split("/")[-1]
        except Exception:
            pass

    if resolved_pmid:
        retracted, detail = _check_pubmed_retraction(resolved_pmid)
        if retracted:
            return True, detail

    if crossref_miss and not resolved_pmid:
        return False, "DOI not in Crossref, no PMID for PubMed fallback"

    return False, ""


def _check_pubmed_retraction(pmid: str) -> tuple[bool, str]:
    """Check PubMed for retraction markers via E-utilities."""
    try:
        r = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "rettype": "xml", "retmode": "xml"},
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as e:
        log.warning("PubMed request failed for PMID %s: %s", pmid, e)
        return False, ""

    if r.status_code != 200:
        return False, ""

    xml = r.text
    # PubMed marks retractions in PublicationType
    if "Retracted Publication" in xml:
        return True, f"PubMed: Retracted Publication (PMID:{pmid})"
    if "RetractionIn" in xml:
        return True, f"PubMed: RetractionIn notice (PMID:{pmid})"
    # Also check CommentsCorrectionsList for retraction notices
    if "RetractionOf" in xml:
        return True, f"PubMed: This is a retraction notice (PMID:{pmid})"

    return False, ""


def check_funding(doi: str) -> tuple[str, list[str]]:
    """Check OpenAlex for funding sources. Returns (classification, funder_names)."""
    try:
        r = httpx.get(
            f"https://api.openalex.org/works/doi:{doi}",
            headers={"User-Agent": _UA},
            timeout=_TIMEOUT,
        )
    except httpx.HTTPError as e:
        log.warning("OpenAlex request failed for %s: %s", doi, e)
        return "unknown", []

    if r.status_code != 200:
        return "unknown", []

    try:
        data = r.json()
    except ValueError:
        return "unknown", []

    # OpenAlex has both top-level "funders" and nested "grants[].funder"
    funder_names = []

    # Top-level funders (most common)
    for f in data.get("funders", []):
        name = f.get("display_name", "")
        if name:
            funder_names.append(name)

    # Nested grants (fallback)
    if not funder_names:
        for g in data.get("grants", []):
            name = g.get("funder_display_name") or ""
            if not name:
                funder_obj = g.get("funder", {})
                name = funder_obj.get("display_name", "") if isinstance(funder_obj, dict) else ""
            if name:
                funder_names.append(name)

    if not funder_names:
        return "unknown", []

    has_industry = any(
        kw in name.lower() for name in funder_names for kw in INDUSTRY_KEYWORDS
    )
    has_gov = any(
        kw in name.lower() for name in funder_names for kw in GOV_KEYWORDS
    )

    if has_industry and has_gov:
        return "mixed", funder_names
    if has_industry:
        return "industry", funder_names
    if has_gov:
        return "government", funder_names
    return "independent", funder_names


def extract_quality_features(text: str) -> dict:
    """Use Gemini Flash Lite to extract study design, sample size, organism, etc."""
    from google import genai
    from google.genai import types

    truncated = text[:2000] if len(text) > 2000 else text

    prompt = (
        "Extract these fields from the paper text. Return JSON only, no markdown.\n"
        "{\n"
        '  "study_design": one of "RCT", "observational", "case-control", "cohort", '
        '"meta-analysis", "case-report", "review", "in-vitro", "animal-study", "other",\n'
        '  "sample_size": integer or null (number of participants/subjects, '
        "NOT number of studies in a meta-analysis),\n"
        '  "is_candidate_gene_study": boolean (true if pre-GWAS-era single gene '
        "association study that does NOT use genome-wide methods),\n"
        '  "organism": one of "human", "animal", "in_vitro", "mixed" '
        "(what organism/model was studied),\n"
        '  "is_meta_analysis": boolean (true if meta-analysis or systematic review)\n'
        "}\n\n"
        f"Paper text:\n{truncated}"
    )

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        raw = response.text or "{}"
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Flash Lite returned non-JSON: %s", response.text[:200] if response.text else "(empty)")
        return {}
    except Exception as e:
        log.warning("Flash Lite extraction failed: %s", e)
        return {}


# -- Main assessment ----------------------------------------------------------

def assess_paper(paper_id: str, db) -> PaperQuality:
    """Run all quality checks on a paper. Caches result in db."""
    # Check cache first
    cached = db.get_paper_quality(paper_id)
    if cached:
        return PaperQuality(**cached)

    paper = db.get_paper(paper_id)
    if paper is None:
        q = PaperQuality(paper_id=paper_id, missing_fields=["paper_not_found"])
        return q

    doi = paper.get("doi")
    abstract = paper.get("abstract", "") or ""
    full_text = paper.get("full_text", "") or ""
    text = abstract or full_text

    q = PaperQuality(
        paper_id=paper_id,
        assessed_at=datetime.now(timezone.utc).isoformat(),
    )

    # -- Retraction check --
    if doi:
        retracted, detail = check_retraction(doi)
        q.retracted = retracted
        q.retraction_detail = detail
        q.checks_run.append("retraction")
    else:
        q.missing_fields.append("doi_missing:retraction_check_skipped")

    # -- Funding check --
    if doi:
        classification, entities = check_funding(doi)
        q.funding_source = classification
        q.funding_entities = entities
        q.checks_run.append("funding")
    else:
        q.missing_fields.append("doi_missing:funding_check_skipped")

    # -- Flash Lite extraction --
    if text:
        features = extract_quality_features(text)
        q.study_design = features.get("study_design", "unknown")
        q.sample_size = features.get("sample_size")
        q.is_candidate_gene = bool(features.get("is_candidate_gene_study", False))
        q.organism = features.get("organism", "unknown")
        q.is_meta_analysis = bool(features.get("is_meta_analysis", False))
        q.is_case_report_only = q.study_design == "case-report"
        q.checks_run.append("flash_lite_extraction")
    else:
        q.missing_fields.append("no_text:feature_extraction_skipped")

    # -- Compute hard vetoes --
    if q.retracted:
        q.veto_reasons.append("RETRACTED")
    if q.is_candidate_gene:
        q.veto_reasons.append("CANDIDATE_GENE")
    if q.organism in ("animal", "in_vitro"):
        q.veto_reasons.append("NON_HUMAN_ONLY")
    if q.is_case_report_only:
        q.veto_reasons.append("CASE_REPORT_ONLY")
    q.vetoed = len(q.veto_reasons) > 0

    # Cache result
    db.update_paper_quality(paper_id, q.to_json())

    return q


def assess_paper_by_doi(doi: str, db) -> PaperQuality | None:
    """Find paper by DOI in db and assess. Returns None if DOI not found."""
    row = db.conn.execute(
        "SELECT paper_id FROM papers WHERE doi = ?", (doi,)
    ).fetchone()
    if not row:
        return None
    return assess_paper(row["paper_id"], db)
