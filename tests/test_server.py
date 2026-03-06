import json
import pytest
import respx
import httpx
from fastmcp import Client
from fastmcp.exceptions import ToolError
from research_mcp.server import create_mcp
from research_mcp.discovery import S2_BASE
from research_mcp.openalex import OA_BASE

FAKE_SEARCH = {
    "total": 1,
    "data": [{
        "paperId": "abc123",
        "title": "Test Paper",
        "abstract": "An abstract.",
        "year": 2024,
        "authors": [{"name": "Alice"}],
        "citationCount": 10,
        "journal": {"name": "Science"},
        "externalIds": {"DOI": "10.1234/test"},
        "openAccessPdf": None,
    }],
}

FAKE_PAPER = FAKE_SEARCH["data"][0]

FAKE_OA_SEARCH = {
    "meta": {"count": 1},
    "results": [{
        "id": "https://openalex.org/W999",
        "doi": "https://doi.org/10.5678/oa-test",
        "title": "OpenAlex Paper",
        "publication_year": 2024,
        "cited_by_count": 5,
        "authorships": [{"author": {"display_name": "Carol"}}],
        "primary_location": {"source": {"display_name": "PLOS ONE"}},
        "abstract_inverted_index": {"A": [0], "test": [1], "abstract": [2]},
        "ids": {"openalex": "https://openalex.org/W999"},
        "open_access": {"oa_url": None},
    }],
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
async def test_search_papers(mcp):
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH)
    )
    async with Client(mcp) as client:
        result = await client.call_tool("search_papers", {"query": "test"})
        data = json.loads(result.content[0].text)
        assert len(data) == 1
        assert data[0]["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_save_and_get_paper(mcp):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        save_result = await client.call_tool("save_paper", {"paper_id": "abc123"})
        save_data = json.loads(save_result.content[0].text)
        assert save_data["saved"] == "Test Paper"

        get_result = await client.call_tool("get_paper", {"paper_id": "abc123"})
        get_data = json.loads(get_result.content[0].text)
        assert get_data["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_list_corpus(mcp):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        result = await client.call_tool("list_corpus", {})
        data = json.loads(result.content[0].text)
        assert len(data) == 1
        assert data[0]["title"] == "Test Paper"


@pytest.mark.anyio
@respx.mock
async def test_export_for_selve(mcp, selve_root):
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        result = await client.call_tool("export_for_selve", {})
        data = json.loads(result.content[0].text)
        assert data["exported"] == 1

    export_path = selve_root / "interpreted" / "research_papers_export.json"
    assert export_path.exists()
    exported = json.loads(export_path.read_text())
    assert len(exported["entries"]) == 1
    assert exported["entries"][0]["source"] == "papers"


@pytest.mark.anyio
@respx.mock
async def test_search_falls_back_to_openalex_on_s2_429(mcp):
    """When S2 returns 429 after retries, OpenAlex is used as fallback."""
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"})
    )
    respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json=FAKE_OA_SEARCH)
    )
    async with Client(mcp) as client:
        result = await client.call_tool("search_papers", {"query": "test"})
        data = json.loads(result.content[0].text)
        assert len(data) == 1
        assert data[0]["title"] == "OpenAlex Paper"
        assert data[0]["paper_id"] == "W999"


@pytest.mark.anyio
@respx.mock
async def test_search_explicit_openalex_backend(mcp):
    """Explicit backend='openalex' skips S2 entirely."""
    s2_route = respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH)
    )
    respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json=FAKE_OA_SEARCH)
    )
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_papers", {"query": "test", "backend": "openalex"}
        )
        data = json.loads(result.content[0].text)
        assert data[0]["title"] == "OpenAlex Paper"
    assert s2_route.call_count == 0


@pytest.mark.anyio
@respx.mock
async def test_search_explicit_s2_no_fallback(mcp):
    """Explicit backend='s2' does not fall back to OpenAlex."""
    respx.get(f"{S2_BASE}/paper/search").mock(
        return_value=httpx.Response(429, headers={"Retry-After": "0"})
    )
    oa_route = respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json=FAKE_OA_SEARCH)
    )
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="rate-limited"):
            await client.call_tool(
                "search_papers", {"query": "test", "backend": "s2"}
            )
    assert oa_route.call_count == 0


@pytest.mark.anyio
async def test_save_and_get_source(mcp):
    async with Client(mcp) as client:
        save_result = await client.call_tool("save_source", {
            "url": "https://example.com/post",
            "title": "Test Post",
            "content": "Some web content here",
        })
        save_data = json.loads(save_result.content[0].text)
        assert save_data["url"] == "https://example.com/post"
        assert save_data["domain"] == "example.com"
        assert save_data["chars"] == len("Some web content here")

        get_result = await client.call_tool("get_source", {
            "url": "https://example.com/post",
        })
        get_data = json.loads(get_result.content[0].text)
        assert get_data["title"] == "Test Post"
        assert get_data["content"] == "Some web content here"


@pytest.mark.anyio
async def test_list_sources(mcp):
    async with Client(mcp) as client:
        await client.call_tool("save_source", {
            "url": "https://a.com/1",
            "title": "Source A",
            "content": "content a",
        })
        await client.call_tool("save_source", {
            "url": "https://b.com/1",
            "title": "Source B",
            "content": "content b",
        })
        result = await client.call_tool("list_sources", {})
        data = json.loads(result.content[0].text)
        assert len(data) == 2
