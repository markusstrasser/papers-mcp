import httpx
import respx

from research_mcp.pubmed import PubMed, EUTILS_BASE

# Two PMIDs → one linkset each (verified ELink behavior with repeated id params).
FAKE_ELINK = {
    "linksets": [
        {
            "dbfrom": "pubmed",
            "ids": ["20301428"],
            "linksetdbs": [
                {"dbto": "gene", "linkname": "pubmed_gene", "links": ["7040", "1080", "2212"]}
            ],
        },
        {
            "dbfrom": "pubmed",
            "ids": ["23193287"],
            "linksetdbs": [],  # no linked genes
        },
    ]
}


def _pubmed():
    return PubMed(client=httpx.Client(base_url=EUTILS_BASE, timeout=5))


@respx.mock
def test_elink_maps_pmid_to_genes():
    route = respx.get(f"{EUTILS_BASE}/elink.fcgi").mock(
        return_value=httpx.Response(200, json=FAKE_ELINK)
    )
    result = _pubmed().elink(["20301428", "23193287"], target_db="gene")
    assert result["target_db"] == "gene"
    assert result["links"]["20301428"] == ["7040", "1080", "2212"]
    assert result["links"]["23193287"] == []
    # Batch of 2 PMIDs resolves in a SINGLE HTTP request (no per-paper loop).
    assert route.call_count == 1


def test_elink_rejects_unknown_db():
    try:
        _pubmed().elink(["123"], target_db="not_a_db")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "target_db" in str(e)


def test_elink_empty_pmids_no_call():
    # No PMIDs → no HTTP request, empty result.
    with respx.mock:
        route = respx.get(f"{EUTILS_BASE}/elink.fcgi").mock(
            return_value=httpx.Response(200, json=FAKE_ELINK)
        )
        result = _pubmed().elink([], target_db="gene")
        assert result == {"target_db": "gene", "links": {}}
        assert route.call_count == 0


@respx.mock
def test_elink_error_body_surfaced():
    respx.get(f"{EUTILS_BASE}/elink.fcgi").mock(
        return_value=httpx.Response(429, text="rate limited")
    )
    try:
        _pubmed().elink(["123"], target_db="gene")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "429" in str(e)
        assert "rate limited" in str(e)


def test_api_key_raises_qps():
    """With an API key the min-interval shrinks (10 qps vs 3)."""
    anon = PubMed()
    keyed = PubMed(api_key="abc")
    assert keyed._min_interval < anon._min_interval
