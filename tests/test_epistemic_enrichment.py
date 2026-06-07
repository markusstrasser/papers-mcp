"""fetch_paper epistemic read-loop enrichment.

fetch_paper carries the corpus epistemic surface (active verdict attestations +
active claim-relations + linear support_balance + conflict) in-band on its
response — mirroring what corpus_lookup already surfaces — so an agent fetching a
paper (about to read/act on it) is SHOWN whether the source is under an active
refutation without a separate corpus_lookup call. The enrichment is best-effort:
any failure leaves the normal fetch result intact.

CORPUS_ROOT is redirected to a per-test tmpdir (the `corpus_root` fixture), so
graph.duckdb reads/writes here NEVER touch the live ~/Projects/corpus. store_root()
and graph_db_path() both resolve CORPUS_ROOT at call time.
"""

import json
import tempfile
from pathlib import Path

import httpx
import pytest
import respx
from fastmcp import Client

from corpus_core.annotate import annotate as corpus_annotate
from corpus_core.ingest import ingest_pdf

from research_mcp.discovery import S2_BASE
from research_mcp.papers import corpus_store
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
    """Redirect the canonical corpus store to a per-test tmpdir.

    Critically isolates graph.duckdb so epistemic_surface(...) reads from the
    tmp store, never the live ~/Projects/corpus.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    monkeypatch.setenv("CORPUS_ROOT", str(root))
    monkeypatch.setenv("RESEARCH_MCP_AUTO_PARSE", "0")
    return root


@pytest.fixture
def mcp(data_dir, selve_root, corpus_root):
    return create_mcp(data_dir=data_dir, selve_root=selve_root)


def _seed_paper(doi: str, text: str) -> str:
    """Pre-populate the store with a tiny PDF + parsed.<parser_id>/page.md and
    return its corpus source_id (store_paper_id)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n%fake pdf bytes for tests\n%%EOF\n")
        pdf_path = Path(f.name)
    meta = ingest_pdf(corpus_store(), pdf_path, doi=doi, skip_parse=True)
    paper_id = meta["paper_id"]
    parsed = corpus_store().paper_path(paper_id) / "parsed.test@1"
    parsed.mkdir(parents=True, exist_ok=True)
    (parsed / "page.md").write_text(text, encoding="utf-8")
    pdf_path.unlink(missing_ok=True)
    return paper_id


def _stub_metadata_routes(doi: str) -> None:
    respx.get(f"{S2_BASE}/paper/abc123").mock(
        return_value=httpx.Response(200, json=FAKE_PAPER)
    )
    respx.get(f"https://api.crossref.org/works/{doi}").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"title": ["Normal paper"], "update-to": [], "relation": {}}
            },
        )
    )
    respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
        return_value=httpx.Response(200, json={"funders": [{"display_name": "NIH"}]})
    )


@pytest.mark.anyio
@respx.mock
async def test_fetch_paper_carries_epistemic_status(mcp, monkeypatch, corpus_root):
    """A fetched paper under an active verdict + active refute relation surfaces
    its epistemic status (conflict=True, the attesting repo, the refuting
    relation) in-band on fetch_paper's response."""
    doi = "10.1234/test"
    extracted = (
        "Association of 5-HTTLPR polymorphism with depression risk in 250 patients. "
        "This open-label observational study found a significant association."
    )
    store_paper_id = _seed_paper(doi, extracted)

    # Seed corpus epistemic state against the fetched source (writes annotations.jsonl
    # AND projects into the tmp graph.duckdb via corpus_core.annotate -> index_annotation).
    # (1) an active verdict attestation:
    corpus_annotate(
        store_paper_id,
        store=corpus_store(),
        repo="genomics",
        actor_type="service",
        actor_id="urn:agent:service:test@0",
        scope="verdict",
        output_uri=f"corpus://{store_paper_id}/verdict",
    )
    # (2) an active refute relation whose OBJECT is this source (=> conflict fires):
    corpus_annotate(
        store_paper_id,
        store=corpus_store(),
        repo="genomics",
        actor_type="service",
        actor_id="urn:agent:service:test@0",
        scope="claim_relation",
        relation={
            "relation_class": "refute",
            "subject_refs": ["corpus:doi_10_9999_refuter"],
            "object_refs": [f"corpus:{store_paper_id}"],
            "detector": "test-detector",
        },
    )

    monkeypatch.setattr("research_mcp.server.download_paper", lambda d: store_paper_id)
    _stub_metadata_routes(doi)

    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        fetch_result = await client.call_tool("fetch_paper", {"paper_id": "abc123"})
        fetch_data = json.loads(fetch_result.content[0].text)

        # Existing contract preserved.
        assert fetch_data["store_paper_id"] == store_paper_id
        assert fetch_data["doi"] == doi
        assert fetch_data["pdf"] == "paper.pdf"

        # New in-band epistemic surface, mirroring corpus_lookup's key shape.
        epi = fetch_data["epistemic_status"]
        assert "active_annotations" in epi
        assert "active_relations" in epi
        assert "epistemic" in epi

        # The active verdict attestation is visible, attributed to genomics.
        assert epi["epistemic"]["active_verdict_count"] >= 1
        assert "genomics" in epi["epistemic"]["attesting_repos"]

        # The refute relation against this source fires the conflict flag.
        assert epi["epistemic"]["conflict"] is True
        assert len(epi["epistemic"]["refuting_relations"]) >= 1
        # Linear support_balance is populated (sign-weighted tally, not a probability).
        assert epi["epistemic"]["support_balance"] is not None
        assert epi["epistemic"]["support_balance"]["n_refute"] >= 1


@pytest.mark.anyio
@respx.mock
async def test_enrichment_failure_is_swallowed(mcp, monkeypatch, corpus_root):
    """If epistemic_surface raises (here: graph_db_path blows up), the failure is
    swallowed — fetch_paper still returns its normal result, just without the
    epistemic_status field. Enrichment must NEVER break a fetch."""
    doi = "10.1234/test"
    extracted = "Some unrelated paper with no corpus epistemic state."
    store_paper_id = _seed_paper(doi, extracted)

    monkeypatch.setattr("research_mcp.server.download_paper", lambda d: store_paper_id)

    # Force a hard failure inside the enrichment block. epistemic_surface is the
    # enrichment entry point on the fetch path; making it raise exercises the
    # swallow-and-continue guard regardless of its internals.
    def _boom(*_args, **_kwargs) -> Path:
        raise RuntimeError("graph db unavailable")

    monkeypatch.setattr("research_mcp.server.epistemic_surface", _boom)
    _stub_metadata_routes(doi)

    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        fetch_result = await client.call_tool("fetch_paper", {"paper_id": "abc123"})
        fetch_data = json.loads(fetch_result.content[0].text)

        # Fetch succeeded with its normal fields despite the enrichment blowup.
        assert "error" not in fetch_data
        assert fetch_data["store_paper_id"] == store_paper_id
        assert fetch_data["doi"] == doi
        assert fetch_data["pdf"] == "paper.pdf"
        # Enrichment was swallowed → no epistemic field on the result.
        assert "epistemic_status" not in fetch_data


@pytest.mark.anyio
@respx.mock
async def test_clean_paper_has_no_conflict(mcp, monkeypatch, corpus_root):
    """A fetched paper with no refuting state still gets an epistemic_status
    field reporting no conflict and no refuting/qualifying relations — proving
    the enrichment runs (and stays quiet) on a clean source. The paper does carry
    the fetch's OWN raw_fetch provenance annotation (written by fetch_paper before
    the read-loop), so active_verdict_count counts that — what matters is that no
    refutation is signalled."""
    doi = "10.1234/test"
    extracted = "A clean paper, no verdicts, no refutations."
    store_paper_id = _seed_paper(doi, extracted)

    monkeypatch.setattr("research_mcp.server.download_paper", lambda d: store_paper_id)
    _stub_metadata_routes(doi)

    async with Client(mcp) as client:
        await client.call_tool("save_paper", {"paper_id": "abc123"})
        fetch_result = await client.call_tool("fetch_paper", {"paper_id": "abc123"})
        fetch_data = json.loads(fetch_result.content[0].text)

        epi = fetch_data["epistemic_status"]
        assert epi["epistemic"]["conflict"] is False
        assert epi["epistemic"]["refuting_relations"] == []
        assert epi["epistemic"]["qualifying_relations"] == []
        assert epi["epistemic"]["support_balance"] is None
