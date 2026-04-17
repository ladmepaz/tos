from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from root_cocitation_similarity import compute_root_citers
from root_combined_similarity import build_combined_outputs, detect_subtopics
from root_structural_similarity import compute_root_influence_sets
from root_tfidf_similarity import (
    build_sap_graph,
    compute_tfidf_vectors,
    extract_root_records,
)

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_cluster_metrics")


def normalize_weights(*weights: float) -> list[float]:
    """Normalize positive weights so they sum to 1."""
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("At least one weight must be positive.")
    return [weight / weight_sum for weight in weights]


def average_pair_strength(
    edges_df: pd.DataFrame,
    members: list[str],
) -> float:
    """Average pair strength among members of the same cluster."""
    if len(members) < 2:
        return 0.0

    member_set = set(members)
    internal_edges = edges_df[
        edges_df["source_root_id"].isin(member_set)
        & edges_df["target_root_id"].isin(member_set)
    ]
    if internal_edges.empty:
        return 0.0
    return float(internal_edges["combined_similarity"].mean())


def average_external_strength(
    edges_df: pd.DataFrame,
    members: list[str],
) -> float:
    """Average pair strength from a cluster to roots outside the cluster."""
    member_set = set(members)
    external_edges = edges_df[
        edges_df["source_root_id"].isin(member_set)
        ^ edges_df["target_root_id"].isin(member_set)
    ]
    if external_edges.empty:
        return 0.0
    return float(external_edges["combined_similarity"].mean())


def compute_cluster_support(
    graph: nx.DiGraph,
    cluster_root_node_ids: list[str],
) -> tuple[float, int, int]:
    """Measure how much of the visible tree connects back to the cluster roots."""
    supporting_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs.get("ToS") in {"trunk", "leaf"}
    ]
    if not supporting_nodes:
        return 0.0, 0, 0

    root_set = set(cluster_root_node_ids)
    supported_count = 0
    for node in supporting_nodes:
        reachable_nodes = nx.descendants(graph, node)
        if reachable_nodes & root_set:
            supported_count += 1

    return supported_count / len(supporting_nodes), supported_count, len(supporting_nodes)


def build_cluster_membership_lookup(subtopics_df: pd.DataFrame) -> dict[str, str]:
    """Map each root_id to its cluster/subtopic id."""
    return dict(zip(subtopics_df["root_id"], subtopics_df["subtopic_id"]))


def compute_root_anchor_scores(
    root_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cluster_lookup: dict[str, str],
) -> pd.DataFrame:
    """Compute root anchor score from SAP rank and within-cluster tie strength."""
    sap_min = float(root_df["sap_rank"].min())
    sap_max = float(root_df["sap_rank"].max())
    sap_range = sap_max - sap_min
    edge_lookup = {
        tuple(sorted((row.source_root_id, row.target_root_id))): row.combined_similarity
        for row in edges_df.itertuples(index=False)
    }

    anchor_rows = []
    for row in root_df.itertuples(index=False):
        cluster_id = cluster_lookup[row.root_id]
        cluster_members = [
            root_id for root_id, member_cluster_id in cluster_lookup.items()
            if member_cluster_id == cluster_id and root_id != row.root_id
        ]
        if cluster_members:
            internal_strength = sum(
                edge_lookup.get(tuple(sorted((row.root_id, other_root_id))), 0.0)
                for other_root_id in cluster_members
            ) / len(cluster_members)
        else:
            internal_strength = 0.0

        if sap_range > 0:
            normalized_sap = (float(row.sap_rank) - sap_min) / sap_range
        else:
            normalized_sap = 1.0

        anchor_score = (0.6 * normalized_sap) + (0.4 * internal_strength)
        anchor_rows.append(
            {
                "subtopic_id": cluster_id,
                "root_id": row.root_id,
                "node_id": row.node_id,
                "sap_rank": row.sap_rank,
                "normalized_sap_rank": round(normalized_sap, 6),
                "within_cluster_pair_strength": round(internal_strength, 6),
                "root_anchor_score": round(anchor_score, 6),
                "title": row.title,
                "year": row.year,
                "journal": row.journal,
                "source_title": row.source_title,
            }
        )

    anchor_df = pd.DataFrame(anchor_rows).sort_values(
        ["subtopic_id", "root_anchor_score", "sap_rank"],
        ascending=[True, False, False],
    )
    return anchor_df


def compute_cluster_metrics(
    graph: nx.DiGraph,
    subtopics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    cohesion_weight: float,
    separation_weight: float,
    support_weight: float,
) -> pd.DataFrame:
    """Compute cohesion, separation, support, and final cluster strength."""
    cluster_rows = []
    for summary_row in summary_df.itertuples(index=False):
        members_df = subtopics_df[subtopics_df["subtopic_id"] == summary_row.subtopic_id]
        member_root_ids = members_df["root_id"].tolist()
        member_node_ids = members_df["node_id"].tolist()

        cohesion = average_pair_strength(edges_df, member_root_ids)
        external_strength = average_external_strength(edges_df, member_root_ids)
        separation = 1.0 - external_strength
        support, supported_nodes, total_support_nodes = compute_cluster_support(
            graph,
            member_node_ids,
        )
        cluster_strength = (
            (cohesion_weight * cohesion)
            + (separation_weight * separation)
            + (support_weight * support)
        )

        cluster_rows.append(
            {
                "subtopic_id": summary_row.subtopic_id,
                "subtopic_label": summary_row.subtopic_label,
                "member_count": int(summary_row.member_count),
                "representative_root_id": summary_row.representative_root_id,
                "representative_node_id": summary_row.representative_node_id,
                "cluster_cohesion": round(cohesion, 6),
                "cluster_external_strength": round(external_strength, 6),
                "cluster_separation": round(separation, 6),
                "cluster_support": round(support, 6),
                "supported_trunk_leaf_count": supported_nodes,
                "total_trunk_leaf_count": total_support_nodes,
                "cluster_strength": round(cluster_strength, 6),
                "top_terms": getattr(summary_row, "top_terms", ""),
                "member_root_ids": summary_row.member_root_ids,
                "member_node_ids": summary_row.member_node_ids,
            }
        )

    cluster_df = pd.DataFrame(cluster_rows).sort_values(
        ["cluster_strength", "cluster_cohesion", "member_count"],
        ascending=[False, False, False],
    )
    return cluster_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute root-cluster metrics from combined similarity outputs."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-text", type=float, default=0.45)
    parser.add_argument("--w-cocitation", type=float, default=0.35)
    parser.add_argument("--w-structural", type=float, default=0.20)
    parser.add_argument("--cohesion-weight", type=float, default=0.45)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--support-weight", type=float, default=0.30)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="Minimum combined similarity to connect two roots in the subtopic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_text, w_cocitation, w_structural = normalize_weights(
        args.w_text,
        args.w_cocitation,
        args.w_structural,
    )
    cohesion_weight, separation_weight, support_weight = normalize_weights(
        args.cohesion_weight,
        args.separation_weight,
        args.support_weight,
    )

    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    tfidf_vectors = compute_tfidf_vectors(records)
    root_citers = compute_root_citers(graph, records)
    influence_sets = compute_root_influence_sets(graph, records)
    root_df, matrix_df, edges_df = build_combined_outputs(
        records,
        tfidf_vectors,
        root_citers,
        influence_sets,
        w_text,
        w_cocitation,
        w_structural,
    )
    _, subtopics_df, summary_df = detect_subtopics(
        records,
        tfidf_vectors,
        edges_df,
        args.min_similarity,
    )

    cluster_lookup = build_cluster_membership_lookup(subtopics_df)
    anchor_df = compute_root_anchor_scores(root_df, edges_df, cluster_lookup)
    cluster_df = compute_cluster_metrics(
        graph,
        subtopics_df,
        summary_df,
        edges_df,
        cohesion_weight,
        separation_weight,
        support_weight,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_df.to_csv(args.output_dir / "root_cluster_metrics.csv", index=False)
    anchor_df.to_csv(args.output_dir / "root_anchor_metrics.csv", index=False)
    edges_df.to_csv(args.output_dir / "root_pair_strength.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_cluster_members.csv", index=False)
    summary_df.to_csv(args.output_dir / "root_cluster_summary.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_combined_matrix.csv", index=False)

    print(f"Root papers: {len(root_df):,}")
    print(f"Clusters: {len(cluster_df):,}")
    print(f"Saved cluster metrics: {(args.output_dir / 'root_cluster_metrics.csv').resolve()}")
    print(f"Saved anchor metrics: {(args.output_dir / 'root_anchor_metrics.csv').resolve()}")
    print(f"Saved pair strengths: {(args.output_dir / 'root_pair_strength.csv').resolve()}")
    print(f"Saved cluster members: {(args.output_dir / 'root_cluster_members.csv').resolve()}")


if __name__ == "__main__":
    main()
