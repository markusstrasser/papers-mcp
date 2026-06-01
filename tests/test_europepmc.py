import httpx
import respx

from research_mcp.europepmc import EuropePMC, EPMC_BASE

FAKE_CORE_RECORD = {
    "id": "41741303",
    "source": "MED",
    "pmid": "41741303",
    "pmcid": "PMC1234567",
    "doi": "10.1016/j.jcf.2026.02.013",
    "title": "Clinical evaluation of CFTR variant I507del.",
    "authorString": "Smith J, Doe A, Roe B.",
    "pubYear": "2026",
    "journalTitle": "Journal of Cystic Fibrosis",
    "citedByCount": 7,
    "isOpenAccess": "N",
    "abstractText": "The CFTR variant I507del was evaluated for function.",
}

FAKE_SEARCH = {"resultList": {"result": [FAKE_CORE_RECORD]}}

FAKE_CITATIONS = {
    "citationList": {
        "citation": [
            {
                "id": "42142142",
                "source": "MED",
                "title": "A citing paper.",
                "pubYear": 2026,
                "citedByCount": 3,
            }
        ]
    }
}

FAKE_REFERENCES = {
    "referenceList": {
        "reference": [
            {"id": "23193287", "source": "MED", "title": "A referenced paper.", "pubYear": 2012}
        ]
    }
}


def _epmc():
    # qps high so the rate gate never sleeps in tests
    return EuropePMC(qps=1000, client=httpx.Client(base_url=EPMC_BASE, timeout=5))


@respx.mock
def test_search_shape():
    respx.get(f"{EPMC_BASE}/search").mock(return_value=httpx.Response(200, json=FAKE_SEARCH))
    results = _epmc().search("CFTR variant", limit=10)
    assert len(results) == 1
    r = results[0]
    assert r["paper_id"] == "EPMC:MED:41741303"
    assert r["doi"] == "10.1016/j.jcf.2026.02.013"
    assert r["title"].startswith("Clinical evaluation")
    assert r["abstract"] == "The CFTR variant I507del was evaluated for function."
    assert r["authors"] == ["Smith J", "Doe A", "Roe B."]
    assert r["year"] == 2026
    assert r["venue"] == "Journal of Cystic Fibrosis"
    assert r["citation_count"] == 7
    assert r["external_ids"]["PubMed"] == "41741303"
    assert r["external_ids"]["PubMedCentral"] == "PMC1234567"
    assert r["external_ids"]["DOI"] == "10.1016/j.jcf.2026.02.013"


@respx.mock
def test_search_empty():
    respx.get(f"{EPMC_BASE}/search").mock(
        return_value=httpx.Response(200, json={"resultList": {"result": []}})
    )
    assert _epmc().search("nonexistent xyz") == []


@respx.mock
def test_citations():
    respx.get(f"{EPMC_BASE}/MED/41741303/citations").mock(
        return_value=httpx.Response(200, json=FAKE_CITATIONS)
    )
    cites = _epmc().citations("MED", "41741303")
    assert len(cites) == 1
    assert cites[0]["paper_id"] == "EPMC:MED:42142142"
    assert cites[0]["citation_count"] == 3
    assert cites[0]["external_ids"]["PubMed"] == "42142142"


@respx.mock
def test_references():
    respx.get(f"{EPMC_BASE}/MED/41741303/references").mock(
        return_value=httpx.Response(200, json=FAKE_REFERENCES)
    )
    refs = _epmc().references("MED", "41741303")
    assert len(refs) == 1
    assert refs[0]["paper_id"] == "EPMC:MED:23193287"


@respx.mock
def test_error_body_surfaced():
    respx.get(f"{EPMC_BASE}/search").mock(
        return_value=httpx.Response(500, text="upstream exploded")
    )
    try:
        _epmc().search("CFTR")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "500" in str(e)
        assert "upstream exploded" in str(e)


def test_rate_gate_sleeps(monkeypatch):
    """Min-interval gate sleeps when calls are closer than the interval."""
    slept = []
    monkeypatch.setattr("research_mcp.europepmc.time.sleep", lambda s: slept.append(s))
    times = iter([100.0, 100.0, 100.1])  # set last_request, then check
    monkeypatch.setattr("research_mcp.europepmc.time.monotonic", lambda: next(times))
    c = EuropePMC(qps=2.0)  # 0.5s min interval
    c._last_request = 100.0
    c._wait()
    assert slept and slept[0] > 0
