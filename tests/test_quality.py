import json

import httpx
import pytest
import respx

from research_mcp.db import PaperDB
from research_mcp.quality import (
    _heuristic_quality_features,
    assess_paper,
    check_retraction,
)


@pytest.fixture
def db(tmp_path):
    return PaperDB(tmp_path / "papers.db")


@respx.mock
def test_check_retraction_detects_title_marker():
    doi = "10.1038/s41586-024-07219-0"
    respx.get(f"https://api.crossref.org/works/{doi}").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "title": ["RETRACTED ARTICLE: The economic commitment of climate change"],
                    "subtype": None,
                    "update-to": [],
                    "relation": {},
                }
            },
        )
    )

    retracted, detail = check_retraction(doi)
    assert retracted is True
    assert "Title marked as retracted" in detail


def test_heuristic_features_flag_candidate_gene_and_non_human():
    text = """
    Association of 5-HTTLPR polymorphism with depression risk in 312 patients and 280 controls.
    This observational study reports a significant association. Experiments in mice were also performed.
    """
    features = _heuristic_quality_features(text)
    assert features["is_candidate_gene_study"] is True
    assert features["organism"] == "mixed"
    assert features["sample_size"] == 312


def test_heuristic_features_do_not_flag_pgx_as_candidate_gene():
    text = """
    Pharmacogenomic study of CYP2D6 genotype and codeine metabolism in 142 patients.
    This open-label trial measured pharmacokinetic outcomes and adverse events.
    """
    features = _heuristic_quality_features(text)
    assert features["is_candidate_gene_study"] is False
    assert features["control_type"] == "unclear"
    assert features["blinding"] == "open-label"


@respx.mock
def test_assess_paper_works_without_model_access(db):
    doi = "10.1234/test"
    db.upsert_paper(
        {
            "paper_id": "paper-1",
            "doi": doi,
            "title": "Association of 5-HTTLPR polymorphism with depression risk",
            "abstract": "Observational association study in 250 patients.",
            "authors": [{"name": "Alice", "affiliations": ["University X"]}],
            "venue": "Nature",
        }
    )
    db.update_paper_pdf(
        "paper-1",
        "/tmp/paper-1.pdf",
        "Association of 5-HTTLPR polymorphism with depression risk in 250 patients. "
        "This observational study found a significant association. Data available on request.",
    )

    respx.get(f"https://api.crossref.org/works/{doi}").mock(
        return_value=httpx.Response(200, json={"message": {"title": ["Normal paper"], "update-to": [], "relation": {}}})
    )
    respx.get(f"https://api.openalex.org/works/doi:{doi}").mock(
        return_value=httpx.Response(200, json={"funders": [{"display_name": "NIH"}]})
    )

    quality = assess_paper("paper-1", db)
    assert quality.retracted is False
    assert quality.is_candidate_gene is True
    assert quality.vetoed is True
    assert "CANDIDATE_GENE" in quality.veto_reasons
    assert quality.funding_source == "government"
    assert quality.data_availability == "on-request"

    cached = db.get_paper_quality("paper-1")
    assert cached is not None
    assert cached["vetoed"] is True
