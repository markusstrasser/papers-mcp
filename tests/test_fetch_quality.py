import json
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp import Client

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
def mcp(data_dir, selve_root):
    return create_mcp(data_dir=data_dir, selve_root=selve_root)


@pytest.mark.anyio
@respx.mock
async def test_fetch_paper_returns_and_persists_quality(mcp, monkeypatch, tmp_path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf bytes")

    monkeypatch.setattr("research_mcp.server.download_paper", lambda doi, pdir: pdf_path)
    monkeypatch.setattr(
        "research_mcp.server.extract_text",
        lambda path: (
            "Association of 5-HTTLPR polymorphism with depression risk in 250 patients. "
            "This open-label observational study found a significant association. "
            "Data available on request."
        ),
    )

    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    respx.get("https://api.crossref.org/works/10.1234/test").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Normal paper"], "update-to": [], "relation": {}}})
    )
    respx.get("https://api.openalex.org/works/doi:10.1234/test").mock(
        return_value=httpx.Response(200, json={"funders": [{"display_name": "NIH"}]})
    )

    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        fetch_result = await client.call_tool("fetch_paper", {"paper_id": "abc123"})
        fetch_data = json.loads(fetch_result.content[0].text)

        assert fetch_data["doi"] == "10.1234/test"
        assert fetch_data["title"] == "Association of 5-HTTLPR polymorphism with depression risk"
        assert fetch_data["quality_status"] == "assessed"
        assert fetch_data["quality"]["is_candidate_gene"] is True
        assert fetch_data["quality"]["vetoed"] is True
        assert "CANDIDATE_GENE" in fetch_data["quality"]["veto_reasons"]

        get_result = await client.call_tool("get_paper", {"paper_id": "abc123"})
        get_data = json.loads(get_result.content[0].text)
        assert get_data["quality_status"] == "assessed"
        assert get_data["quality"]["is_candidate_gene"] is True
