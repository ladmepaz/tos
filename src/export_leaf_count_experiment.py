from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_citation_network import (
    build_citation_network,
    enrich_network_with_article_data,
    load_article_metadata,
    load_citation_edges,
    remove_self_loops,
)
from sap import LEAF, SAP, Sap


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/leaf_count_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export experimental SAP leaf lists for different max_leaves values."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-leaves", type=int, default=50)
    parser.add_argument("--experiment-leaves", type=int, default=100)
    return parser.parse_args()


def build_enriched_graph(citation_csv: Path, articles_csv: Path):
    citation_df = load_citation_edges(citation_csv)
    article_df = load_article_metadata(articles_csv)
    graph = build_citation_network(citation_df)
    graph = remove_self_loops(graph)
    graph = Sap.clean_graph(graph)
    return enrich_network_with_article_data(graph, article_df)


def run_sap(graph, max_leaves: int):
    sap = Sap(max_leaves=max_leaves)
    return sap.tree(graph)


def extract_leaves(graph, max_leaves: int) -> pd.DataFrame:
    records = []
    leaves = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if attrs.get(LEAF, 0) > 0
    ]
    leaves = sorted(
        leaves,
        key=lambda item: (item[1].get(LEAF, 0), item[1].get(SAP, 0), str(item[0])),
        reverse=True,
    )

    for rank, (node, attrs) in enumerate(leaves, start=1):
        records.append(
            {
                "leaf_rank": rank,
                "max_leaves_run": max_leaves,
                "node_id": node,
                "title": attrs.get("title") or "",
                "abstract": attrs.get("abstract") or "",
                "year": attrs.get("year"),
                "journal": attrs.get("journal") or "",
                "source_title": attrs.get("source_title") or "",
                "doi": attrs.get("doi") or "",
                "has_article_metadata": attrs.get("has_article_metadata", False),
                "leaf_connection_score": attrs.get(LEAF, 0),
                "sap_rank": attrs.get(SAP, 0),
                "internal_indegree": graph.in_degree(node),
                "internal_outdegree": graph.out_degree(node),
            }
        )
    return pd.DataFrame(records)


def add_experiment_flags(
    baseline_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_nodes = set(baseline_df["node_id"])
    flagged_df = experiment_df.copy()
    flagged_df["in_baseline_50"] = flagged_df["node_id"].isin(baseline_nodes)
    flagged_df["is_added_by_100_leaf_run"] = ~flagged_df["in_baseline_50"]
    added_df = flagged_df[flagged_df["is_added_by_100_leaf_run"]].copy()
    return flagged_df, added_df


def summarize_leaves(df: pd.DataFrame, label: str) -> dict:
    year_series = pd.to_numeric(df["year"], errors="coerce")
    return {
        "run": label,
        "leaf_count": len(df),
        "metadata_count": int(df["has_article_metadata"].sum()),
        "metadata_share": round(float(df["has_article_metadata"].mean()), 4) if len(df) else 0,
        "min_year": int(year_series.min()) if year_series.notna().any() else None,
        "median_year": float(year_series.median()) if year_series.notna().any() else None,
        "max_year": int(year_series.max()) if year_series.notna().any() else None,
        "min_leaf_connection_score": float(df["leaf_connection_score"].min()) if len(df) else 0,
        "median_leaf_connection_score": float(df["leaf_connection_score"].median()) if len(df) else 0,
        "max_leaf_connection_score": float(df["leaf_connection_score"].max()) if len(df) else 0,
    }


def main() -> None:
    args = parse_args()
    graph = build_enriched_graph(args.citations, args.articles)
    baseline_graph = run_sap(graph, args.baseline_leaves)
    experiment_graph = run_sap(graph, args.experiment_leaves)

    baseline_df = extract_leaves(baseline_graph, args.baseline_leaves)
    experiment_df = extract_leaves(experiment_graph, args.experiment_leaves)
    experiment_df, added_df = add_experiment_flags(baseline_df, experiment_df)
    summary_df = pd.DataFrame(
        [
            summarize_leaves(baseline_df, f"top_{args.baseline_leaves}"),
            summarize_leaves(experiment_df, f"top_{args.experiment_leaves}"),
            summarize_leaves(added_df, f"added_{args.baseline_leaves + 1}_{args.experiment_leaves}"),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / f"leaves_top_{args.baseline_leaves}.csv"
    experiment_path = args.output_dir / f"leaves_top_{args.experiment_leaves}.csv"
    added_path = args.output_dir / f"leaves_added_{args.baseline_leaves + 1}_{args.experiment_leaves}.csv"
    summary_path = args.output_dir / "leaf_count_experiment_summary.csv"
    baseline_df.to_csv(baseline_path, index=False)
    experiment_df.to_csv(experiment_path, index=False)
    added_df.to_csv(added_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Baseline leaves: {len(baseline_df):,}")
    print(f"Experiment leaves: {len(experiment_df):,}")
    print(f"Added by experiment: {len(added_df):,}")
    print(f"Saved baseline leaves: {baseline_path.resolve()}")
    print(f"Saved experiment leaves: {experiment_path.resolve()}")
    print(f"Saved added leaves: {added_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
