"""Paper PDF download (Sci-Hub + OA) and full-text extraction.

The canonical corpus store at ~/Projects/corpus/ owns PDFs and parsed text.
This module is a thin shim: it fetches bytes from the network, hands them to
`corpus_core.store` for ingestion, and returns a `paper_id`. Downstream code
resolves `paper_id` → on-disk path via `corpus_core.store.paper_path()`.

Extraction pipeline (in order):
  1. Marker output: parsed/paper.md if ingested with parse
  2. Gemini Flash-Lite: upload PDF → structured markdown
  3. PyMuPDF raw text (last-resort offline fallback)
"""

import logging
import os
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from corpus_core import store as ps
from corpus_core.ingest import ingest_pdf
from corpus_core.store import CorpusStore

logger = logging.getLogger(__name__)


def corpus_store() -> CorpusStore:
    """Process-boundary corpus handle.

    corpus_core uses explicit store injection (no module-level default root);
    research-mcp resolves the root at its boundary. Root = CORPUS_ROOT env var,
    else the canonical ~/Projects/corpus. NOT cached — CORPUS_ROOT can change
    in-process (tests redirect it per-case); a cached handle would go stale and
    leak one root into another context. Construction is trivial.
    """
    root = os.environ.get("CORPUS_ROOT")
    return CorpusStore(
        Path(root).expanduser() if root else Path("~/Projects/corpus").expanduser()
    )


SCIHUB_MIRRORS = [
    "https://sci-hub.ru",
    "https://sci-hub.ee",
    "https://sci-hub.st",
]

_UA = {"User-Agent": "Mozilla/5.0"}


def _auto_parse_enabled() -> bool:
    """RESEARCH_MCP_AUTO_PARSE=0 disables marker invocation (smoke/test path)."""
    return os.environ.get("RESEARCH_MCP_AUTO_PARSE", "1") not in (
        "0",
        "false",
        "False",
        "",
    )


def download_paper(doi: str) -> Optional[str]:
    """Download a paper by DOI and ingest into the canonical store.

    Returns the `paper_id` (e.g. ``doi_10_1234_test``) on success, ``None`` on
    download failure. Cache-hit returns instantly without touching the network.
    """
    paper_id = ps.derive_paper_id(doi=doi)

    # Cache check — if already in the store, return without downloading.
    if corpus_store().exists(paper_id):
        return paper_id

    with tempfile.TemporaryDirectory(prefix="research-mcp-dl-") as td:
        td_path = Path(td)
        safe_name = re.sub(r"[^\w\-.]", "_", doi) + ".pdf"
        dest = td_path / safe_name

        pdf_path: Optional[Path] = None
        arxiv_pdf = _arxiv_pdf_url(doi)
        if arxiv_pdf:  # arXiv: go straight to the real PDF (Sci-Hub/doi.org can't serve it)
            pdf_path = _download_url(arxiv_pdf, dest)
        if not pdf_path:
            for base_url in SCIHUB_MIRRORS:
                pdf_path = _try_scihub(doi, base_url, dest)
                if pdf_path:
                    break
        if not pdf_path:
            pdf_path = _download_url(f"https://doi.org/{doi}", dest)
        if not pdf_path:
            return None

        return _ingest(pdf_path, doi=doi)


# Patterns to derive DOI/PMID from common preprint URLs.
_DOI_URL_PATTERNS = [
    # bioRxiv / medRxiv: https://www.medrxiv.org/content/10.1101/2026.04.10.26350624v1.full.pdf
    re.compile(r"(?:biorxiv|medrxiv)\.org/content/(10\.1101/[^/?#v]+)", re.I),
    # Generic doi.org redirects
    re.compile(r"doi\.org/(10\.[^/?#\s]+/[^?#\s]+)", re.I),
]
_ARXIV_PATTERN = re.compile(
    r"arxiv\.org/(?:pdf|abs)/([\w.\-]+?)(?:v\d+)?(?:\.pdf)?(?:[?#]|$)", re.I
)
_PMID_PATTERN = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.I)
_ARXIV_DOI_PATTERN = re.compile(r"10\.48550/arxiv\.([\w.\-]+)", re.I)


def _arxiv_pdf_url(doi: Optional[str]) -> Optional[str]:
    """arXiv papers aren't on Sci-Hub and doi.org redirects to the abs PAGE (HTML, not a PDF),
    so the generic chain fails on every arXiv ID (recurring gotcha — researchers fell back to
    ar5iv HTML by hand). The real PDF is one URL away; construct it directly."""
    m = _ARXIV_DOI_PATTERN.search(doi or "")
    return f"https://arxiv.org/pdf/{m.group(1)}" if m else None


def _derive_id_from_url(url: str) -> dict:
    """Best-effort DOI/PMID extraction from a URL. Returns kwargs for derive_paper_id."""
    for pat in _DOI_URL_PATTERNS:
        m = pat.search(url)
        if m:
            return {"doi": m.group(1).rstrip(".")}
    m = _ARXIV_PATTERN.search(url)
    if m:
        return {"doi": f"10.48550/arXiv.{m.group(1)}"}
    m = _PMID_PATTERN.search(url)
    if m:
        return {"pmid": m.group(1)}
    return {}


def download_url(url: str, name: Optional[str] = None) -> Optional[str]:
    """Download a paper PDF from a direct URL and ingest into the store.

    Auto-derives DOI/PMID from common preprint URLs. Falls back to a
    sha-prefix paper_id when the URL is opaque.

    Returns the `paper_id` on success, ``None`` on download failure.
    """
    derived = _derive_id_from_url(url)
    # Fast path: if we can derive a paper_id from the URL alone and it's
    # already in the store, skip the network.
    if derived:
        try:
            cached_id = ps.derive_paper_id(**derived)
            if corpus_store().exists(cached_id):
                return cached_id
        except ps.PaperStoreError:
            pass

    with tempfile.TemporaryDirectory(prefix="research-mcp-dl-") as td:
        td_path = Path(td)
        safe_name = name or re.sub(r"[^\w\-.]", "_", url.split("/")[-1] or "paper")
        if not safe_name.endswith(".pdf"):
            safe_name += ".pdf"
        dest = td_path / safe_name

        # An arXiv abs/pdf URL -> canonical /pdf/ (an /abs/ URL returns HTML, not a PDF).
        fetch_url = _arxiv_pdf_url(derived.get("doi")) or url
        pdf_path = _download_url(fetch_url, dest)
        if not pdf_path:
            return None

        # If we couldn't derive identity from the URL, fall back to pdf_sha.
        if not derived:
            pdf_sha = ps.sha256_file(pdf_path)
            derived = {"pdf_sha": pdf_sha}

        return _ingest(pdf_path, **derived)


def _ingest(pdf_path: Path, **id_fields) -> Optional[str]:
    """Hand a freshly-downloaded PDF to the canonical store. Returns paper_id."""
    # ingest_pdf computes pdf_sha256 itself and derives the id from it when no
    # doi/pmid is supplied, so a stale pdf_sha kwarg is both unaccepted and
    # redundant — drop it.
    id_fields.pop("pdf_sha", None)
    try:
        meta = ingest_pdf(
            corpus_store(), pdf_path, skip_parse=not _auto_parse_enabled(), **id_fields
        )
        return meta["paper_id"]
    except Exception as e:
        logger.warning("ingest_pdf failed for %s: %s", pdf_path.name, e)
        return None


def extract_text(paper_id: str) -> str:
    """Return full-text markdown for ``paper_id``.

    Resolution order (Phase 1.5 of substrate-migration plan):
      1. Any ``parsed.<parser_id>/page.md`` from a local extractor (mineru,
         pymupdf4llm, trafilatura) — first found wins.
      2. corpus_core.extract.pdf_llm fallback (Gemini Flash-Lite).
      3. PyMuPDF raw text — last-resort offline.
    """
    rec = corpus_store().get(paper_id)  # raises PaperNotFoundError on miss

    # 1. Any local parsed.<parser_id>/page.md
    for child in sorted(rec.path.iterdir()):
        if child.is_dir() and child.name.startswith("parsed."):
            md = child / "page.md"
            if md.exists():
                text = md.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text

    pdf_path = rec.pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"No paper.pdf for {paper_id} at {pdf_path}")

    # 2. Gemini fallback via corpus_core (sole owner of LLM extraction now)
    try:
        from corpus_core.extract import pdf_llm

        result = pdf_llm.extract(pdf_path)
        if len(result.parsed_markdown) >= 100:
            return result.parsed_markdown
        logger.warning("LLM fallback returned too little text for %s", paper_id)
    except Exception as e:
        logger.warning("LLM fallback failed for %s: %s", paper_id, e)

    # 3. PyMuPDF raw text
    return _extract_with_pymupdf(pdf_path)


def _extract_with_pymupdf(pdf_path: Path) -> str:
    """Last-resort: raw text extraction via pymupdf."""
    import fitz

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _try_scihub(doi: str, base_url: str, dest: Path) -> Optional[Path]:
    """Try downloading from a single Sci-Hub mirror."""
    try:
        url = f"{base_url}/{doi}"
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        match = re.search(r'citation_pdf_url"\s+content="([^"]+)"', html)
        if not match:
            match = re.search(
                r'(?:src|href|data)\s*=\s*["\']?(/(?:download|storage)/[^"\'>\s]+\.pdf[^"\'>\s]*)',
                html,
            )
        if not match:
            match = re.search(
                r'(?:src|href)\s*=\s*["\']?((?:https?:)?//[^"\'>\s]+\.pdf[^"\'>\s]*)',
                html,
            )
        if not match:
            return None

        pdf_url = match.group(1)
        if pdf_url.startswith("/"):
            pdf_url = base_url + pdf_url
        elif pdf_url.startswith("//"):
            pdf_url = "https:" + pdf_url

        req = urllib.request.Request(pdf_url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)

        if dest.stat().st_size > 1000:
            return dest
        dest.unlink(missing_ok=True)

    except Exception as e:
        logger.warning("Sci-Hub %s failed for %s: %s", base_url, doi, e)

    return None


def _download_url(url: str, dest: Path) -> Optional[Path]:
    """Download a URL to dest. Returns dest if successful."""
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type and "octet" not in content_type:
                return None
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        if dest.stat().st_size > 1000:
            return dest
        dest.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("Download failed for %s: %s", url, e)
    return None
