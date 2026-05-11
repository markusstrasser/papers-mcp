#!/usr/bin/env python3
"""Phase 0 measurement report — duplicate-fetch rate over a time window.

Reads research-mcp's `fetch_log` table and reports the empirical premise
behind the cross-attestation substrate decision (decisions/2026-05-11-cross-
attestation-substrate.md, in ~/Projects/agent-infra).

Headline metric: of all fetch_paper invocations in the window, what fraction
hit a source we had ALREADY fully fetched (full_text already populated)?

  - If ≥10%: the cache-inside-fetch_paper work (Phase 2) is justified.
  - If <10%: the premise is wrong; the substrate is over-engineering.

Usage:
  uv run python3 scripts/analyze_fetch_log.py                 # last 7 days
  uv run python3 scripts/analyze_fetch_log.py --days 14
  uv run python3 scripts/analyze_fetch_log.py --db custom.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _ok(msg): print(f"  ✓ {msg}")
def _warn(msg): print(f"  ! {msg}")
def _header(s): print(f"\n[{s}]")
def _kv(k, v): print(f"  {k:30s} {v}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(Path(__file__).resolve().parents[1] / "data" / "papers.db"))
    p.add_argument("--days", type=int, default=7)
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: {db_path} does not exist")
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Existence check — table may not be populated yet
    cols = [r[1] for r in conn.execute("PRAGMA table_info(fetch_log)").fetchall()]
    if not cols:
        print("ERROR: fetch_log table missing — is this an upgraded research-mcp DB?")
        return 1

    window = f"datetime('now', '-{args.days} days')"

    _header(f"Phase 0 fetch-opportunity report — last {args.days} days")
    _kv("DB", db_path)

    total = conn.execute(f"SELECT COUNT(*) FROM fetch_log WHERE requested_at >= {window}").fetchone()[0]
    _kv("Total fetch_paper invocations", total)
    if total == 0:
        _warn("No data in window. Call fetch_paper at least once or extend --days.")
        return 0

    # By status
    _header("By result status")
    for row in conn.execute(
        f"SELECT result_status, COUNT(*) AS n FROM fetch_log "
        f"WHERE requested_at >= {window} GROUP BY result_status ORDER BY n DESC"
    ).fetchall():
        _kv(row["result_status"], row["n"])

    # By caller
    _header("By caller_tag (set via $RESEARCH_MCP_CALLER per repo)")
    caller_rows = conn.execute(
        f"SELECT caller_tag, COUNT(*) AS n FROM fetch_log "
        f"WHERE requested_at >= {window} GROUP BY caller_tag ORDER BY n DESC"
    ).fetchall()
    for row in caller_rows:
        _kv(row["caller_tag"], row["n"])
    if len(caller_rows) == 1 and caller_rows[0]["caller_tag"] == "unknown":
        _warn("All fetches tagged 'unknown' — set RESEARCH_MCP_CALLER per repo .mcp.json env to get cross-repo overlap.")

    # The headline metric
    _header("Headline metric — duplicate fetches")
    hits = conn.execute(
        f"SELECT COUNT(*) FROM fetch_log "
        f"WHERE had_full_text_before = 1 AND requested_at >= {window}"
    ).fetchone()[0]
    pct = (hits * 100.0 / total) if total else 0.0
    _kv("Fetches where full_text was already present", f"{hits} ({pct:.1f}%)")

    if pct >= 10:
        _ok("≥10% — cache-inside-fetch_paper (Phase 2) is justified")
    elif pct >= 3:
        _warn(f"{pct:.1f}% — borderline. Extend the measurement window or look at caller breakdown.")
    else:
        _warn(f"{pct:.1f}% — below threshold. Substrate work is likely over-engineering.")

    # Top duplicates by source_id_norm
    _header("Top duplicated source_ids (re-fetched ≥ 2×)")
    dupes = conn.execute(
        f"""SELECT source_id_norm, COUNT(*) AS n,
                   SUM(had_full_text_before) AS dup_hits
            FROM fetch_log
            WHERE requested_at >= {window} AND source_id_norm IS NOT NULL
            GROUP BY source_id_norm
            HAVING n >= 2
            ORDER BY n DESC LIMIT 20"""
    ).fetchall()
    if not dupes:
        _warn("None — no source_id was fetched more than once in the window.")
    else:
        for row in dupes:
            _kv(row["source_id_norm"][:60], f"{row['n']} fetches, {row['dup_hits']} hit cached")

    # Cross-caller overlap (the bridge ROI question)
    _header("Cross-caller overlap (same source fetched by different repos)")
    if len(caller_rows) > 1:
        cross = conn.execute(
            f"""SELECT source_id_norm, COUNT(DISTINCT caller_tag) AS callers,
                       GROUP_CONCAT(DISTINCT caller_tag) AS who
                FROM fetch_log
                WHERE requested_at >= {window} AND source_id_norm IS NOT NULL
                GROUP BY source_id_norm
                HAVING callers >= 2
                ORDER BY callers DESC LIMIT 20"""
        ).fetchall()
        if not cross:
            _warn("No source_ids fetched by ≥2 different callers in the window.")
            _warn("This is the cross-repo bridge ROI signal — zero overlap = no bridge value.")
        else:
            for row in cross:
                _kv(row["source_id_norm"][:60], f"{row['callers']} callers: {row['who']}")
    else:
        _warn("Only one caller_tag in window — cross-caller overlap measurement requires per-repo tags.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
