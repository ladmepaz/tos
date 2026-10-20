"""Apply explicitly requested, DOI-verified corrections to missing counts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def apply_citation_corrections(articles: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Fill missing cited_by only; never replace reported zeros or other counts.

    SR and DOI must both match. Existing matching counts make this idempotent;
    conflicts fail instead of silently mixing a different citation snapshot.
    """
    corrections = json.loads(path.read_text(encoding="utf-8"))
    result = articles.copy()
    seen = set()
    for record in corrections:
        sr = record["SR"]
        count = record["cited_by"]
        if sr in seen or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Invalid or duplicate citation correction: {sr}")
        seen.add(sr)
        matches = result["SR"].astype(str).str.strip().eq(sr)
        if not matches.any():
            raise ValueError(f"Citation correction SR not found: {sr}")
        doi_matches = result.loc[matches, "doi"].fillna("").astype(str).str.strip().str.casefold()
        if not doi_matches.eq(record["doi"].strip().casefold()).all():
            raise ValueError(f"Citation correction DOI mismatch: {sr}")
        existing = result.loc[matches, "cited_by"]
        missing = existing.isna() | existing.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(existing, errors="coerce")
        if ((~missing) & (numeric.isna() | numeric.ne(count))).any():
            raise ValueError(f"Citation correction conflicts with an existing count: {sr}")
        result.loc[existing.index[missing], "cited_by"] = count
    return result
