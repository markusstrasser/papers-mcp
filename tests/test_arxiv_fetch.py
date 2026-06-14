"""arXiv fetch path: download_paper must go straight to the real arxiv.org/pdf URL.

Regression guard for the recurring gotcha where fetch_paper failed on EVERY arXiv ID —
Sci-Hub doesn't carry arXiv and doi.org redirects to the abs PAGE (HTML, not a PDF), so the
generic chain always failed and researchers fell back to ar5iv HTML by hand.
"""
import research_mcp.papers as papers


def test_arxiv_pdf_url():
    assert papers._arxiv_pdf_url("10.48550/arXiv.2406.02373") == "https://arxiv.org/pdf/2406.02373"
    assert papers._arxiv_pdf_url("10.48550/arxiv.2510.04051") == "https://arxiv.org/pdf/2510.04051"
    # non-arXiv DOIs are untouched (no regression)
    assert papers._arxiv_pdf_url("10.1101/2026.04.10.26350624") is None
    assert papers._arxiv_pdf_url("10.1234/foo.bar") is None
    assert papers._arxiv_pdf_url(None) is None


def test_download_paper_tries_arxiv_pdf_first(monkeypatch):
    calls = []

    def fake_dl(url, dest):
        calls.append(url)
        return dest if "arxiv.org/pdf" in url else None

    monkeypatch.setattr(papers, "_download_url", fake_dl)
    monkeypatch.setattr(papers, "_try_scihub", lambda *a, **k: None)  # Sci-Hub never has arXiv
    monkeypatch.setattr(papers, "_ingest", lambda path, **k: "arxiv_paper_id")
    monkeypatch.setattr(papers, "corpus_store",
                        lambda: type("S", (), {"exists": lambda self, x: False})())

    pid = papers.download_paper("10.48550/arXiv.2406.02373")
    assert pid == "arxiv_paper_id"
    # the FIRST network attempt is the real arXiv PDF, before any Sci-Hub/doi.org fallback
    assert calls and calls[0] == "https://arxiv.org/pdf/2406.02373", calls
