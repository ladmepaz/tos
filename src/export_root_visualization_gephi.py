from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_NODES_INPUT = Path("outputs/root_visualization/root_visualization_metrics.csv")
DEFAULT_PAIR_INPUT = Path("outputs/root_cluster_metrics/root_pair_strength.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_visualization/gephi")


def build_gephi_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Create a Gephi-friendly node table for the selected root clusters."""
    gephi_nodes = pd.DataFrame(
        {
            "Id": nodes_df["node_id"],
            "Label": nodes_df["title"].fillna("").where(
                nodes_df["title"].fillna("").str.strip() != "",
                nodes_df["node_id"],
            ),
            "size": nodes_df["size"].round(6),
            "color": nodes_df["color"],
            "citations_received": nodes_df["citations_received"],
            "log_citations_received": nodes_df["log_citations_received"].round(6),
            "citation_velocity": nodes_df["citation_velocity"].round(6),
            "log_citation_velocity": nodes_df["log_citation_velocity"].round(6),
            "citation_velocity_score": nodes_df["citation_velocity_score"].round(6),
            "global_size_score": nodes_df["global_size_score"].round(6),
            "local_size_score": nodes_df["local_size_score"].round(6),
            "final_root_size_score": nodes_df["final_root_size_score"].round(6),
            "compressed_root_size_score": nodes_df["compressed_root_size_score"].round(6),
            "visual_size_level": nodes_df["visual_size_level"],
            "subtopic_id": nodes_df["subtopic_id"],
            "subtopic_label": nodes_df["subtopic_label"],
            "cluster_order": nodes_df["cluster_order"],
            "cluster_strength": nodes_df["cluster_strength"].round(6),
            "cluster_cohesion": nodes_df["cluster_cohesion"].round(6),
            "cluster_separation": nodes_df["cluster_separation"].round(6),
            "cluster_support": nodes_df["cluster_support"].round(6),
            "is_representative": nodes_df["is_representative"],
            "root_anchor_score": nodes_df["root_anchor_score"].round(6),
            "sap_rank": nodes_df["sap_rank"],
            "year": nodes_df["year"],
            "journal": nodes_df["journal"].fillna(""),
            "source_title": nodes_df["source_title"].fillna(""),
            "has_article_metadata": nodes_df["has_article_metadata"],
        }
    )
    return gephi_nodes


def build_gephi_edges(nodes_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Create an optional Gephi edge table among the selected roots."""
    selected_ids = set(nodes_df["root_id"])
    filtered_pairs = pair_df[
        pair_df["source_root_id"].isin(selected_ids)
        & pair_df["target_root_id"].isin(selected_ids)
    ].copy()
    filtered_pairs = filtered_pairs.sort_values(
        "combined_similarity",
        ascending=False,
    )
    gephi_edges = pd.DataFrame(
        {
            "Source": filtered_pairs["source_node_id"],
            "Target": filtered_pairs["target_node_id"],
            "Type": "Undirected",
            "Weight": filtered_pairs["combined_similarity"].round(6),
            "text_similarity": filtered_pairs["text_similarity"].round(6),
            "co_citation_similarity": filtered_pairs["co_citation_similarity"].round(6),
            "structural_similarity": filtered_pairs["structural_similarity"].round(6),
            "combined_similarity": filtered_pairs["combined_similarity"].round(6),
        }
    )
    return gephi_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export root visualization tables for Gephi."
    )
    parser.add_argument("--nodes-input", type=Path, default=DEFAULT_NODES_INPUT)
    parser.add_argument("--pair-input", type=Path, default=DEFAULT_PAIR_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.nodes_input)
    pair_df = pd.read_csv(args.pair_input)

    gephi_nodes = build_gephi_nodes(nodes_df)
    gephi_edges = build_gephi_edges(nodes_df, pair_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gephi_nodes.to_csv(args.output_dir / "root_nodes_gephi.csv", index=False)
    gephi_edges.to_csv(args.output_dir / "root_edges_gephi.csv", index=False)

    print(f"Nodes: {len(gephi_nodes):,}")
    print(f"Edges: {len(gephi_edges):,}")
    print(f"Saved nodes: {(args.output_dir / 'root_nodes_gephi.csv').resolve()}")
    print(f"Saved edges: {(args.output_dir / 'root_edges_gephi.csv').resolve()}")


if __name__ == "__main__":
    main()
