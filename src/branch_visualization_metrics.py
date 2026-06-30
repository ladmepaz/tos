from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path

import pandas as pd

from root_tfidf_similarity import build_sap_graph
from trunk_visualization_metrics import (
    interpolate_color,
    normalize_series,
    split_hex_color,
)


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_BRANCH_ROLES = Path("outputs/branch_member_roles/branch_member_roles.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/branch_visualization")
BRANCH_COLOR_LOW = "#C4AE97"
BRANCH_COLOR_HIGH = "#441A03"


def compute_branch_sizes(
    graph,
    branch_df: pd.DataFrame,
    min_size: float,
    max_size: float,
) -> pd.DataFrame:
    """Compute branch node sizes from incoming citations."""
    sized_df = branch_df.copy()
    sized_df["citations_received"] = (
        sized_df["node_id"].map(graph.in_degree).fillna(0).astype(float)
    )
    sized_df["log_citations_received"] = sized_df["citations_received"].apply(
        math.log1p
    )
    sized_df["citation_size_score"] = normalize_series(
        sized_df["log_citations_received"]
    )
    # Mild compression keeps low-cited recent branch papers visible.
    sized_df["compressed_size_score"] = sized_df["citation_size_score"].apply(
        math.sqrt
    )
    sized_df["size"] = min_size + (
        sized_df["compressed_size_score"] * (max_size - min_size)
    )
    return sized_df


def compute_citation_velocity(
    branch_df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """Compute citation velocity and assign the shared brown ToS gradient."""
    velocity_df = branch_df.copy()
    velocity_df["publication_year"] = pd.to_numeric(
        velocity_df["year"],
        errors="coerce",
    ).fillna(current_year)
    velocity_df["years_since_publication"] = (
        current_year - velocity_df["publication_year"] + 1
    ).clip(lower=1)
    velocity_df["citation_velocity"] = (
        velocity_df["citations_received"] / velocity_df["years_since_publication"]
    )
    velocity_df["log_citation_velocity"] = velocity_df["citation_velocity"].apply(
        math.log1p
    )
    velocity_df["citation_velocity_score"] = normalize_series(
        velocity_df["log_citation_velocity"]
    )
    velocity_df["color"] = velocity_df["citation_velocity_score"].apply(
        lambda score: interpolate_color(BRANCH_COLOR_LOW, BRANCH_COLOR_HIGH, score)
    )
    rgb_values = velocity_df["color"].apply(split_hex_color)
    velocity_df["color_r"] = rgb_values.apply(lambda rgb: rgb[0])
    velocity_df["color_g"] = rgb_values.apply(lambda rgb: rgb[1])
    velocity_df["color_b"] = rgb_values.apply(lambda rgb: rgb[2])
    return velocity_df


def enrich_with_graph_attributes(graph, branch_df: pd.DataFrame) -> pd.DataFrame:
    """Add SAP and ToS attributes from the current citation graph."""
    enriched_df = branch_df.copy()
    enriched_df["sap_rank"] = enriched_df["node_id"].map(
        lambda node_id: graph.nodes[node_id].get("sap_rank", 0)
        if node_id in graph
        else 0
    )
    enriched_df["baseline_ToS"] = enriched_df["node_id"].map(
        lambda node_id: graph.nodes[node_id].get("ToS", "")
        if node_id in graph
        else ""
    )
    enriched_df["in_current_graph"] = enriched_df["node_id"].map(
        lambda node_id: node_id in graph
    )
    return enriched_df


def build_branch_visualization_metrics(
    citations: Path,
    articles: Path,
    branch_roles: Path,
    min_size: float,
    max_size: float,
    current_year: int,
) -> pd.DataFrame:
    """Build branch-only size and color metrics for SVG/Gephi visualization."""
    graph = build_sap_graph(citations, articles)
    branch_df = pd.read_csv(branch_roles)
    branch_df = enrich_with_graph_attributes(graph, branch_df)
    branch_df = compute_branch_sizes(graph, branch_df, min_size, max_size)
    branch_df = compute_citation_velocity(branch_df, current_year)
    branch_df["branch_order"] = (
        branch_df["branch_label"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    )
    role_order = {
        "core": 1,
        "peripheral": 2,
        "background_methodological": 3,
        "missing_metadata": 4,
    }
    branch_df["role_order"] = (
        branch_df["branch_member_role"].map(role_order).fillna(5).astype(int)
    )
    branch_df = branch_df.sort_values(
        [
            "branch_order",
            "role_order",
            "branch_core_score",
            "year",
            "citations_received",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    return branch_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build branch-only visualization metrics for ToS branch papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--branch-roles", type=Path, default=DEFAULT_BRANCH_ROLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-size", type=float, default=6.0)
    parser.add_argument("--max-size", type=float, default=25.0)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    branch_df = build_branch_visualization_metrics(
        args.citations,
        args.articles,
        args.branch_roles,
        args.min_size,
        args.max_size,
        args.current_year,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "branch_visualization_metrics.csv"
    branch_df.to_csv(metrics_path, index=False)

    print(f"Branch nodes: {len(branch_df):,}")
    print(
        "Size range: "
        f"{branch_df['size'].min():.2f} to {branch_df['size'].max():.2f}"
    )
    print(
        "Citation velocity range: "
        f"{branch_df['citation_velocity'].min():.4f} to "
        f"{branch_df['citation_velocity'].max():.4f}"
    )
    print(f"Saved branch visualization metrics: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
