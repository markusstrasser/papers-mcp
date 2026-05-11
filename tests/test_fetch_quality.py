"""fetch_paper end-to-end test against the canonical corpus store.

Tests the (paper_id-returning) download_paper / download_url / extract_text
signatures.
"""
import json
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp import Client

from corpus_core import store as ps
from corpus_core.ingest import ingest_pdf

from research_mcp.discovery import S2_BASE
from research_mcp.server import create_mcp


FAKE_PAPER = {
    "paperId": "abc123",
    "title": "Association of 5-HTTLPR polymorphism with depression risk",
    "abstract": "Observational association study in 250 patients.",
    "year": 2024,
    "authors": [{"name": "Alice"}],
    "citationCount": 10,
    "journal": {"name": "Science"},
    "externalIds": {"DOI": "10.1234/test"},
    "openAccessPdf": None,
}


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def selve_root(tmp_path):
    interpreted = tmp_path / "selve" / "interpreted"
    interpreted.mkdir(parents=True)
    return tmp_path / "selve"


@pytest.fixture
def corpus_root(tmp_path, monkeypatch):
    """Redirect the canonical corpus store to a per-test tmpdir."""
    root = tmp_path / "corpus"
    root.mkdir()
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    # AUTO_PARSE=0 keeps marker out of unit tests
    monkeypatch.setenv("RESEARCH_MCP_AUTO_PARSE", "0")
    return root


@pytest.fixture
def mcp(data_dir, selve_root, corpus_root):
    return create_mcp(data_dir=data_dir, selve_root=selve_root)


def _seed_paper(doi: str, text: str) -> str:
    """Pre-populate the store with a tiny PDF + parsed.<parser_id>/page.md."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n%fake pdf bytes for tests\n%%EOF\n")
        pdf_path = Path(f.name)
    meta = ingest_pdf(pdf_path, doi=doi, skip_parse=True)
    paper_id = meta["paper_id"]
    # Write parsed.test@1/page.md so extract_text resolves on the local path
    # (Phase 1.5 immutable parsed dirs).
    parsed = ps.paper_path(paper_id) / "parsed.test@1"
    parsed.mkdir(parents=True, exist_ok=True)
    (parsed / "page.md").write_text(text, encoding="utf-8")
    pdf_path.unlink(missing_ok=True)
    return paper_id


@pytest.mark.anyio
@respx.mock
async def test_fetch_paper_returns_and_persists_quality(mcp, monkeypatch, corpus_root):
    doi = "10.1234/test"
    extracted = (
        "Association of 5-HTTLPR polymorphism with depression risk in 250 patients. "
        "This open-label observational study found a significant association. "
        "Data available on request."
    )
    expected_paper_id = _seed_paper(doi, extracted)

    # Stub download_paper to return the cached paper_id (cache-hit path).
    # The real implementation also returns instantly when the store contains
    # this paper_id, but stubbing keeps the test offline.
    monkeypatch.setattr(
        "research_mcp.server.download_paper",
        lambda d: expected_paper_id,
    )

    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    respx.get(f"https://api.crossref.org/works/{doi}").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Normal paper"], "update-to": [], "relation": {}}})
    )
    respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
        return_value=httpx.Response(200, json={"funders": [{"display_name": "NIH"}]})
    )

    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        fetch_result = await client.call_tool("fetch_paper", {"paper_id": "abc123"})
        fetch_data = json.loads(fetch_result.content[0].text)

        assert fetch_data["doi"] == doi
        assert fetch_data["store_paper_id"] == expected_paper_id
        assert fetch_data["pdf"] == "paper.pdf"
        assert fetch_data["title"] == "Association of 5-HTTLPR polymorphism with depression risk"
        assert fetch_data["quality_status"] == "assessed"
        assert fetch_data["quality"]["is_candidate_gene"] is True
        assert fetch_data["quality"]["vetoed"] is True
        assert "CANDIDATE_GENE" in fetch_data["quality"]["veto_reasons"]

        get_result = await client.call_tool("get_paper", {"paper_id": "abc123"})
        get_data = json.loads(get_result.content[0].text)
        assert get_data["quality_status"] == "assessed"
        assert get_data["quality"]["is_candidate_gene"] is True
