from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path

import pandas as pd

from root_cluster_metrics import (
    build_cluster_membership_lookup,
    compute_cluster_metrics,
    compute_root_anchor_scores,
    normalize_weights,
)
from root_cocitation_similarity import compute_root_citers
from root_combined_similarity import (
    DEFAULT_NTF_SIMILARITY,
    build_combined_outputs,
    detect_subtopics,
    load_ntf_similarity_lookup,
)
from root_structural_similarity import compute_root_influence_sets
from root_tfidf_similarity import (
    build_sap_graph,
    compute_tfidf_vectors,
    extract_root_records,
)

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_visualization")
ROOT_COLOR_LOW = "#C4AE97"
ROOT_COLOR_HIGH = "#441A03"
ROOT_SIZE_LEVELS = [14.0, 18.0, 22.0, 26.0, 30.0]


def normalize_series(values: pd.Series) -> pd.Series:
    """Normalize a numeric series to [0, 1] with a safe constant fallback."""
    if values.empty:
        return values.astype(float)

    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(1.0, index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


def compute_root_sizes(
    graph,
    roots_df: pd.DataFrame,
    local_weight: float,
    global_weight: float,
    min_size: float,
    max_size: float,
) -> pd.DataFrame:
    """Compute root-only visualization sizes from indegree."""
    sized_df = roots_df.copy()
    sized_df["citations_received"] = sized_df["node_id"].map(graph.in_degree).astype(float)
    sized_df["log_citations_received"] = sized_df["citations_received"].apply(math.log1p)
    sized_df["global_size_score"] = normalize_series(sized_df["log_citations_received"])
    sized_df["local_size_score"] = (
        sized_df.groupby("subtopic_id", group_keys=False)["log_citations_received"]
        .apply(normalize_series)
        .astype(float)
    )
    sized_df["final_root_size_score"] = (
        (local_weight * sized_df["local_size_score"])
        + (global_weight * sized_df["global_size_score"])
    )
    sized_df["compressed_root_size_score"] = (
        0.5 * sized_df["final_root_size_score"]
        + 0.5 * sized_df["final_root_size_score"].apply(math.sqrt)
    )
    # Use discrete editorial size classes instead of a fully continuous scale.
    rank_fraction = normalize_series(sized_df["compressed_root_size_score"])
    level_edges = [0.2, 0.4, 0.6, 0.8]
    sized_df["visual_size_level"] = rank_fraction.apply(
        lambda value: 1
        + sum(value > edge for edge in level_edges)
    )
    size_levels = ROOT_SIZE_LEVELS
    sized_df["size"] = sized_df["visual_size_level"].apply(
        lambda level: size_levels[int(level) - 1]
    )
    return sized_df


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert an RGB tuple to a hex color."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def interpolate_color(start_hex: str, end_hex: str, score: float) -> str:
    """Linearly interpolate between two hex colors."""
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)
    clamped_score = max(0.0, min(1.0, float(score)))
    mixed_rgb = tuple(
        round(start + ((end - start) * clamped_score))
        for start, end in zip(start_rgb, end_rgb)
    )
    return rgb_to_hex(mixed_rgb)


def compute_citation_velocity(
    roots_df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """Compute citation velocity and a root color gradient."""
    velocity_df = roots_df.copy()
    velocity_df["publication_year"] = pd.to_numeric(
        velocity_df["year"],
        errors="coerce",
    )
    velocity_df["publication_year"] = velocity_df["publication_year"].fillna(current_year)
    velocity_df["years_since_publication"] = (
        current_year - velocity_df["publication_year"] + 1
    ).clip(lower=1)
    velocity_df["citation_velocity"] = (
        velocity_df["citations_received"] / velocity_df["years_since_publication"]
    )
    velocity_df["log_citation_velocity"] = velocity_df["citation_velocity"].apply(math.log1p)
    velocity_df["citation_velocity_score"] = normalize_series(
        velocity_df["log_citation_velocity"]
    )
    velocity_df["color"] = velocity_df["citation_velocity_score"].apply(
        lambda score: interpolate_color(ROOT_COLOR_LOW, ROOT_COLOR_HIGH, score)
    )
    return velocity_df


def select_display_clusters(cluster_df: pd.DataFrame, max_clusters: int) -> list[str]:
    """Keep only the strongest multi-paper clusters for root visualization."""
    eligible_clusters = cluster_df[cluster_df["member_count"] > 1].copy()
    eligible_clusters = eligible_clusters.sort_values(
        ["cluster_strength", "member_count"],
        ascending=[False, False],
    )
    return eligible_clusters["subtopic_id"].head(max_clusters).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build root-only visualization metrics for the strongest root clusters."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-text", type=float, default=0.45)
    parser.add_argument("--w-cocitation", type=float, default=0.35)
    parser.add_argument("--w-structural", type=float, default=0.20)
    parser.add_argument("--w-ntf", type=float, default=0.0)
    parser.add_argument("--ntf-similarity", type=Path, default=DEFAULT_NTF_SIMILARITY)
    parser.add_argument("--cohesion-weight", type=float, default=0.45)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--support-weight", type=float, default=0.30)
    parser.add_argument("--min-similarity", type=float, default=0.35)
    parser.add_argument("--max-clusters", type=int, default=3)
    parser.add_argument("--local-weight", type=float, default=0.70)
    parser.add_argument("--global-weight", type=float, default=0.30)
    parser.add_argument("--min-size", type=float, default=14.0)
    parser.add_argument("--max-size", type=float, default=22.0)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_text, w_cocitation, w_structural, w_ntf = normalize_weights(
        args.w_text,
        args.w_cocitation,
        args.w_structural,
        args.w_ntf,
    )
    cohesion_weight, separation_weight, support_weight = normalize_weights(
        args.cohesion_weight,
        args.separation_weight,
        args.support_weight,
    )
    local_weight, global_weight = normalize_weights(
        args.local_weight,
        args.global_weight,
    )

    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    tfidf_vectors = compute_tfidf_vectors(records)
    root_citers = compute_root_citers(graph, records)
    influence_sets = compute_root_influence_sets(graph, records)
    ntf_similarity_lookup = load_ntf_similarity_lookup(args.ntf_similarity)
    root_df, _, edges_df = build_combined_outputs(
        records,
        tfidf_vectors,
        root_citers,
        influence_sets,
        w_text,
        w_cocitation,
        w_structural,
        w_ntf,
        ntf_similarity_lookup,
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

    selected_cluster_ids = select_display_clusters(cluster_df, args.max_clusters)
    selected_clusters_df = cluster_df[
        cluster_df["subtopic_id"].isin(selected_cluster_ids)
    ].copy()
    selected_clusters_df["cluster_order"] = range(1, len(selected_clusters_df) + 1)

    roots_df = subtopics_df[subtopics_df["subtopic_id"].isin(selected_cluster_ids)].copy()
    roots_df = roots_df.merge(
        anchor_df[
            [
                "subtopic_id",
                "root_id",
                "node_id",
                "normalized_sap_rank",
                "within_cluster_pair_strength",
                "root_anchor_score",
            ]
        ],
        on=["subtopic_id", "root_id", "node_id"],
        how="left",
    )
    roots_df = roots_df.merge(
        selected_clusters_df[
            [
                "subtopic_id",
                "subtopic_label",
                "cluster_strength",
                "cluster_cohesion",
                "cluster_separation",
                "cluster_support",
                "cluster_order",
            ]
        ],
        on=["subtopic_id", "subtopic_label"],
        how="left",
    )
    roots_df = compute_root_sizes(
        graph,
        roots_df,
        local_weight,
        global_weight,
        args.min_size,
        args.max_size,
    )
    roots_df = compute_citation_velocity(
        roots_df,
        args.current_year,
    )
    roots_df = roots_df.sort_values(
        ["cluster_order", "root_anchor_score", "citations_received"],
        ascending=[True, False, False],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots_df.to_csv(args.output_dir / "root_visualization_metrics.csv", index=False)
    selected_clusters_df.to_csv(
        args.output_dir / "root_visualization_clusters.csv",
        index=False,
    )

    print(f"Displayed clusters: {len(selected_cluster_ids):,}")
    print(f"Displayed roots: {len(roots_df):,}")
    print(
        f"Saved root visualization metrics: "
        f"{(args.output_dir / 'root_visualization_metrics.csv').resolve()}"
    )
    print(
        f"Saved root visualization clusters: "
        f"{(args.output_dir / 'root_visualization_clusters.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
