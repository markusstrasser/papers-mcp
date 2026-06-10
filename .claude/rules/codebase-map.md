---
description: Auto-generated file map with cross-file relationships. Updated daily.
paths:
  - "scripts/**"
---
# Codebase Map

<!-- Gov-ID: rule:codebase-map
goal: compact code map for agent navigation (generated)
verifier: null
blast_radius: style
-->

# 17 Python files — generated 2026-06-06
# Edge annotations: → imports  ← imported-by-N-files

## src/research_mcp/

  __init__.py        ← 26 files
  cag.py           Cache-Augmented Generation — stuff full papers into Gem
  cc_ranks.py      Common Crawl domain-rank lookup — cheap authority signa
  db.py            SQLite store for paper metadata and response cache.
  deep_research.py Gemini Deep Research — autonomous multi-step web resear
  discovery.py     Semantic Scholar API client with cachi…  → research_mcp
  europepmc.py     EuropePMC backend — biomedical literature search + cita
  exa_verify.py    Exa /answer-based claim verification.
  extraction.py    Structured extraction tables — Elicit-style column-base
  middleware.py    MCP telemetry middleware for research-mcp.
  openalex.py      OpenAlex API client — fallback for Sem…  → research_mcp
  papers.py        Paper PDF download (Sci-Hub + OA) and full-text extract
  preprints.py     bioRxiv / medRxiv preprint search with date filtering a
  pubmed.py        PubMed (NCBI E-utilities) backend — ELink cross-databas
  quality.py       Evidence quality assessment — mechanical checks, not co
  rcs.py           RCS (Relevance-scoring, Chunking, Summarization) for ev
  server.py        Research MCP server — paper discovery,…  → research_mcp
