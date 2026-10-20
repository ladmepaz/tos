"""Replay the citation-only audit from its frozen pre-correction baseline."""
from pathlib import Path
import hashlib
import json
import math
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fruit_candidates import classify_fruit_signal, normalize_series


def fingerprints():
    paths = [ROOT / "data/processed/bibfusion/All_Articles_tidy.csv"]
    paths += list((ROOT / "outputs/fruits").glob("*.csv"))
    paths += [ROOT / "outputs/fruit_visualization/fruits_top3.csv"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def recalculate(frame):
    df = frame.copy()
    df["years_since_publication"] = (2026 - df["year"] + 1).clip(lower=1)
    df["external_citation_velocity"] = df.cited_by / df.years_since_publication
    df["external_citation_score"] = normalize_series(df.cited_by.apply(math.log1p))
    df["internal_citation_score"] = normalize_series(df.internal_indegree.apply(math.log1p))
    df["external_velocity_score"] = normalize_series(df.external_citation_velocity.apply(math.log1p))
    df["external_attention_gap_raw"] = (df.external_citation_score - df.internal_citation_score).clip(lower=0)
    df["external_attention_gap_score"] = normalize_series(df.external_attention_gap_raw)
    df["fruit_score"] = (0.35 * df.external_velocity_score + 0.35 * df.external_attention_gap_score
                         + 0.20 * df.network_proximity_score + 0.10 * df.text_relevance_score)
    df["fruit_signal_type"] = df.apply(classify_fruit_signal, axis=1)
    return df.sort_values(["fruit_score", "external_citation_velocity", "cited_by"], ascending=False)


def underabsorbed(df):
    return df[df.fruit_signal_type.eq("external_fruit_underabsorbed")]


def main():
    hashes = fingerprints()
    original = pd.read_csv(HERE / "baseline_candidates.csv")
    assert len(original) == 120 and original.SR.is_unique
    assert original.in_current_graph.dtype == bool
    baseline = recalculate(original)
    original_indexed = original.set_index("SR")
    numeric = ["years_since_publication", "external_citation_velocity", "external_citation_score",
               "internal_citation_score", "external_attention_gap_raw", "external_attention_gap_score", "fruit_score"]
    errors = {c: float((baseline.set_index("SR")[c] - original_indexed[c]).abs().max()) for c in numeric}
    assert max(errors.values()) < 1e-12, errors
    assert baseline.set_index("SR").fruit_signal_type.eq(original_indexed.fruit_signal_type).all()
    saved_top = pd.read_csv(ROOT / "outputs/fruit_visualization/fruits_top3.csv")
    assert list(underabsorbed(baseline).head(3).SR) == list(saved_top.SR)

    recovery = json.loads((ROOT / "data/corrections/fruit_citation_recovery_2026.json").read_text(encoding="utf-8"))
    assert len(recovery) == 10
    integrated = pd.read_csv(ROOT / "data/raw/bibfusion/All_Articles.csv", low_memory=False)
    modified = original.copy()
    for r in recovery:
        mask = modified.SR.eq(r["SR"])
        assert mask.sum() == 1
        assert modified.loc[mask, "cited_by"].eq(0).all()
        assert modified.loc[mask, "doi"].str.lower().eq(r["doi"].lower()).all()
        source_rows = integrated[integrated.SR.eq(r["SR"])]
        assert len(source_rows) and pd.to_numeric(source_rows.cited_by, errors="coerce").isna().all()
        modified.loc[mask, "cited_by"] = r["cited_by"]
    corrected = recalculate(modified)
    corrected.to_csv(HERE / "candidates_recalculated.csv", index=False)
    underabsorbed(corrected).to_csv(HERE / "underabsorbed_recalculated.csv", index=False)
    underabsorbed(corrected).head(3).to_csv(HERE / "top3_recalculated.csv", index=False)

    fields = ["cited_by", "fruit_score", "fruit_signal_type", "external_attention_gap_score"]
    comparison = original[["SR", "title", "doi", "year", "internal_indegree"]].copy()
    for label, frame in [("original", baseline), ("recalculated", corrected)]:
        indexed = frame.set_index("SR")
        for field in fields:
            comparison[f"{field}_{label}"] = comparison.SR.map(indexed[field])
        ranks = {sr: i for i, sr in enumerate(underabsorbed(frame).SR, 1)}
        comparison[f"underabsorbed_rank_{label}"] = comparison.SR.map(ranks).astype("Int64")
    comparison["citations_recovered"] = comparison.SR.isin([r["SR"] for r in recovery])
    comparison = comparison.sort_values("fruit_score_recalculated", ascending=False)
    comparison.to_csv(HERE / "comparison_all120.csv", index=False)
    comparison[comparison.citations_recovered].to_csv(HERE / "comparison_recovered10.csv", index=False)

    summary = {"reference_year": 2026, "citation_export_date": "2026-03-07", "population": 120,
               "recovered_records": 10, "weights": [0.35, 0.35, 0.20, 0.10],
               "baseline_max_errors": errors,
               "underabsorbed_original": len(underabsorbed(baseline)),
               "underabsorbed_recalculated": len(underabsorbed(corrected)),
               "top3_original": list(underabsorbed(baseline).head(3).SR),
               "top3_recalculated": list(underabsorbed(corrected).head(3).SR),
               "scope": "Fixed saved candidate population and graph/text features; recover ten missing counts only.",
               "limitations": "Mixed WoS Z9 and Scopus coverage retained; not a full pipeline rebuild or validation.",
               "baseline_sha256": hashlib.sha256((HERE / "baseline_candidates.csv").read_bytes()).hexdigest(),
               "canonical_file_sha256_at_replay": hashes}
    assert fingerprints() == hashes, "An original output or input changed during the experiment"
    summary["original_outputs_unchanged"] = True
    (HERE / "audit_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    columns = ["SR", "cited_by", "internal_indegree", "external_attention_gap_score", "fruit_score"]
    print("RECALCULATED TOP 3")
    print(underabsorbed(corrected).head(3)[columns].to_string(index=False))
    print("RECOVERED 10")
    print(comparison[comparison.citations_recovered].drop(columns=["title", "doi"]).to_string(index=False))
    print("PREVIOUS TOP 3")
    print(comparison[comparison.SR.isin(saved_top.SR)].drop(columns=["title", "doi"]).to_string(index=False))
    print("Baseline errors:", errors)
    print("Underabsorbed:", summary["underabsorbed_original"], "->", summary["underabsorbed_recalculated"])
    print("Unchanged original files:", len(hashes))


if __name__ == "__main__":
    main()
