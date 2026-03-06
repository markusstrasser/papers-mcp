import pytest
import respx
import httpx
from research_mcp.db import PaperDB
from research_mcp.openalex import OpenAlex, OA_BASE, _reconstruct_abstract

FAKE_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.1038/s41586-020-2649-2",
    "title": "Scaling deep learning for materials discovery",
    "display_name": "Scaling deep learning for materials discovery",
    "publication_year": 2020,
    "cited_by_count": 450,
    "authorships": [
        {"author": {"display_name": "Alice Chen"}},
        {"author": {"display_name": "Bob Kim"}},
    ],
    "primary_location": {
        "source": {"display_name": "Nature"},
    },
    "abstract_inverted_index": {
        "Deep": [0],
        "learning": [1, 8],
        "has": [2],
        "emerged": [3],
        "as": [4],
        "a": [5],
        "powerful": [6],
        "tool": [7],
    },
    "ids": {
        "openalex": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.1038/s41586-020-2649-2",
        "pmid": "https://pubmed.ncbi.nlm.nih.gov/12345",
    },
    "open_access": {
        "is_oa": True,
        "oa_url": "https://example.com/oa.pdf",
    },
}

FAKE_SEARCH_RESPONSE = {
    "meta": {"count": 1},
    "results": [FAKE_WORK],
}


@pytest.fixture
def db(tmp_path):
    return PaperDB(tmp_path / "test.db")


@pytest.fixture
def oa(db):
    return OpenAlex(db)


def test_reconstruct_abstract():
    inv = {"The": [0], "cat": [1], "sat": [2], "on": [3], "the": [4], "mat": [5]}
    assert _reconstruct_abstract(inv) == "The cat sat on the mat"


def test_reconstruct_abstract_none():
    assert _reconstruct_abstract(None) is None
    assert _reconstruct_abstract({}) is None


def test_reconstruct_abstract_with_repeated_words():
    inv = {"Deep": [0], "learning": [1, 8], "has": [2], "emerged": [3],
           "as": [4], "a": [5], "powerful": [6], "tool": [7]}
    result = _reconstruct_abstract(inv)
    assert result == "Deep learning has emerged as a powerful tool learning"


@respx.mock
def test_search(oa):
    respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH_RESPONSE)
    )
    results = oa.search("deep learning materials")
    assert len(results) == 1
    r = results[0]
    assert r["paper_id"] == "W2741809807"
    assert r["doi"] == "10.1038/s41586-020-2649-2"
    assert r["title"] == "Scaling deep learning for materials discovery"
    assert r["authors"] == ["Alice Chen", "Bob Kim"]
    assert r["year"] == 2020
    assert r["venue"] == "Nature"
    assert r["citation_count"] == 450
    assert r["open_access_url"] == "https://example.com/oa.pdf"
    assert r["external_ids"]["DOI"] == "10.1038/s41586-020-2649-2"
    assert r["external_ids"]["PubMed"] == "12345"
    assert r["abstract"] is not None
    assert r["abstract"].startswith("Deep learning")


@respx.mock
def test_search_caches(oa):
    route = respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json=FAKE_SEARCH_RESPONSE)
    )
    oa.search("deep learning")
    oa.search("deep learning")  # cache hit
    assert route.call_count == 1


@respx.mock
def test_get_paper(oa):
    respx.get(f"{OA_BASE}/works/W2741809807").mock(
        return_value=httpx.Response(200, json=FAKE_WORK)
    )
    paper = oa.get_paper("W2741809807")
    assert paper["paper_id"] == "W2741809807"
    assert paper["doi"] == "10.1038/s41586-020-2649-2"
    assert paper["citation_count"] == 450


@respx.mock
def test_get_paper_by_doi(oa):
    doi = "10.1038/s41586-020-2649-2"
    respx.get(f"{OA_BASE}/works/https://doi.org/{doi}").mock(
        return_value=httpx.Response(200, json=FAKE_WORK)
    )
    paper = oa.get_paper(doi)
    assert paper["paper_id"] == "W2741809807"


@respx.mock
def test_get_paper_not_found(oa):
    respx.get(f"{OA_BASE}/works/W9999999999").mock(
        return_value=httpx.Response(404)
    )
    assert oa.get_paper("W9999999999") is None


@respx.mock
def test_search_empty(oa):
    respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json={"meta": {"count": 0}, "results": []})
    )
    assert oa.search("nonexistent query xyz") == []


@respx.mock
def test_normalize_missing_fields(oa):
    """Papers with missing optional fields shouldn't crash."""
    minimal_work = {
        "id": "https://openalex.org/W123",
        "title": "Minimal Paper",
        "publication_year": 2024,
        "cited_by_count": 0,
        "authorships": [],
        "ids": {},
    }
    respx.get(f"{OA_BASE}/works").mock(
        return_value=httpx.Response(200, json={"results": [minimal_work]})
    )
    results = oa.search("minimal")
    assert len(results) == 1
    r = results[0]
    assert r["paper_id"] == "W123"
    assert r["title"] == "Minimal Paper"
    assert r["doi"] is None
    assert r["abstract"] is None
    assert r["authors"] == []
    assert r["venue"] is None
    assert r["open_access_url"] is None
