"""Common Crawl domain-rank lookup — cheap authority signal for web sources.

Reads ``~/.cache/cc-domain-ranks/latest.parquet``, populated by meta's
``just cc-ranks-refresh``. Returns None when the cache is missing — callers
should degrade gracefully (the feature is advisory, not required).

The ranks file stores ``host_rev`` in reverse-label form (``com.google``).
CC only publishes DOMAIN-level ranks (not host-level), so we strip to the
registered domain before the lookup. A small 2-part-TLD table handles the
common ccTLDs (``.co.uk``, ``.com.au``, ...); rarer suffixes fall back to
last-2-labels, which is good enough for source grading.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

PARQUET_PATH = Path.home() / ".cache/cc-domain-ranks/latest.parquet"

# Common 2-part public suffixes. Not exhaustive — covers the ccTLDs most
# likely to appear in research sources. For the long tail we fall back to
# last-2-labels, which under-counts subdomain authority but never crashes.
_TWO_PART_TLDS = frozenset({
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.au", "org.au", "edu.au", "gov.au",
    "co.nz", "org.nz",
    "co.jp", "or.jp", "ac.jp", "go.jp",
    "com.br", "org.br",
    "co.za", "org.za",
    "com.cn", "org.cn",
    "co.in", "org.in",
    "com.mx",
    "com.ar",
    "com.tr",
    "com.sg",
    "com.hk",
})


def registered_domain(netloc_or_url: str) -> str | None:
    """Strip a URL or netloc to its registered (eTLD+1) domain, lowercased.

    Returns None for IPs, localhost, or unparseable input.
    """
    s = netloc_or_url.strip().lower()
    if "://" in s:
        s = urlparse(s).netloc
    # strip port + userinfo
    s = s.split("@")[-1].split(":")[0].rstrip(".")
    if not s or s == "localhost":
        return None
    # crude IPv4 check
    if s.replace(".", "").isdigit():
        return None
    labels = s.split(".")
    if len(labels) < 2:
        return None
    last_two = ".".join(labels[-2:])
    if last_two in _TWO_PART_TLDS and len(labels) >= 3:
        return ".".join(labels[-3:])
    return last_two


def _reverse(domain: str) -> str:
    return ".".join(reversed(domain.split(".")))


def authority_tier(harmonic_pos: int | None) -> str:
    """Bucket harmonic centrality rank into a 4-tier authority label.

    These cutoffs are calibrated against the CC ranks distribution: pos 1-10K
    are household-name domains, 10K-100K are well-known sites/publications,
    100K-1M are long-tail but findable, >1M is effectively unknown.
    """
    if harmonic_pos is None:
        return "unranked"
    if harmonic_pos <= 10_000:
        return "top10k"
    if harmonic_pos <= 100_000:
        return "top100k"
    if harmonic_pos <= 1_000_000:
        return "top1m"
    return "tail"


@lru_cache(maxsize=4096)
def lookup_domain(domain_or_url: str) -> dict | None:
    """Look up a domain's rank. Returns None if cache missing or domain absent.

    Result dict keys: ``domain``, ``harmonic_pos``, ``harmonic_val``,
    ``pagerank_pos``, ``n_hosts``, ``tier``.
    """
    if not PARQUET_PATH.exists():
        return None
    domain = registered_domain(domain_or_url)
    if domain is None:
        return None
    try:
        import duckdb  # lazy import — optional dep
    except ImportError:
        log.warning("duckdb not installed; cc_ranks lookup unavailable")
        return None
    rev = _reverse(domain)
    con = duckdb.connect(":memory:")
    try:
        row = con.execute(
            "SELECT harmonicc_pos, harmonicc_val, pr_pos, n_hosts "
            "FROM read_parquet(?) WHERE host_rev = ? LIMIT 1",
            [str(PARQUET_PATH), rev],
        ).fetchone()
    finally:
        con.close()
    if not row:
        return {
            "domain": domain,
            "harmonic_pos": None,
            "harmonic_val": None,
            "pagerank_pos": None,
            "n_hosts": 0,
            "tier": "tail",
        }
    harmonic_pos, harmonic_val, pr_pos, n_hosts = row
    return {
        "domain": domain,
        "harmonic_pos": harmonic_pos,
        "harmonic_val": harmonic_val,
        "pagerank_pos": pr_pos,
        "n_hosts": n_hosts,
        "tier": authority_tier(harmonic_pos),
    }
