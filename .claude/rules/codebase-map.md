---
description: Auto-generated file map with cross-file relationships. Updated daily.
---
# Codebase Map
# 8 Python files — generated 2026-03-13
# Navigation: repo_callgraph(target="name") finds callers across files

## src/research_mcp/

  __init__.py     ← 11 files
  cag.py        Cache-Augmented Generation — stuff full papers into Gem
  db.py         SQLite store for paper metadata and response cache.
  discovery.py  Semantic Scholar API client with cachi…  → research_mcp
  exa_verify.py Exa /answer-based claim verification.
  openalex.py   OpenAlex API client — fallback for Sem…  → research_mcp
  papers.py     Paper PDF download (Sci-Hub + OA) and full-text extract
  server.py     Research MCP server — paper discovery,…  → research_mcp
