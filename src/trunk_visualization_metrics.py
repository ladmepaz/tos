from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path

import pandas as pd

from root_combined_similarity import normalize_weights
from trunk_combined_similarity import (
    DEFAULT_ARTICLES,
    DEFAULT_CITATIONS,
    build_combined_outputs,
    compute_tfidf_vectors,
    compute_trunk_citer_sets,
    compute_trunk_reference_sets,
    detect_subtopics,
    extract_trunk_records,
)
from root_tfidf_similarity import build_sap_graph


DEFAULT_OUTPUT_DIR = Path("outputs/trunk_visualization")
TRUNK_COLOR_LOW = "#C4AE97"
TRUNK_COLOR_HIGH = "#441A03"


def normalize_series(values: pd.Series) -> pd.Series:
    """Normalize a numeric series to [0, 1] with a safe constant fallback."""
    if values.empty:
        return values.astype(float)

    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(1.0, index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


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


def split_hex_color(hex_color: str) -> tuple[int, int, int]:
    """Return RGB channels for Gephi-friendly color imports."""
    return hex_to_rgb(hex_color)


def compute_trunk_sizes(
    graph,
    trunk_df: pd.DataFrame,
    min_size: float,
    max_size: float,
) -> pd.DataFrame:
    """Compute trunk-only node sizes from incoming citations."""
    sized_df = trunk_df.copy()
    sized_df["citations_received"] = sized_df["node_id"].map(graph.in_degree).astype(float)
    sized_df["log_citations_received"] = sized_df["citations_received"].apply(math.log1p)
    sized_df["citation_size_score"] = normalize_series(
        sized_df["log_citations_received"]
    )
    # Square-root compression keeps low-cited trunk papers visible while preserving rank.
    sized_df["compressed_size_score"] = sized_df["citation_size_score"].apply(math.sqrt)
    sized_df["size"] = min_size + (
        sized_df["compressed_size_score"] * (max_size - min_size)
    )
    return sized_df


def compute_citation_velocity(
    trunk_df: pd.DataFrame,
    current_year: int,
) -> pd.DataFrame:
    """Compute citation velocity and assign the trunk brown color gradient."""
    velocity_df = trunk_df.copy()
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
        lambda score: interpolate_color(TRUNK_COLOR_LOW, TRUNK_COLOR_HIGH, score)
    )
    rgb_values = velocity_df["color"].apply(split_hex_color)
    velocity_df["color_r"] = rgb_values.apply(lambda rgb: rgb[0])
    velocity_df["color_g"] = rgb_values.apply(lambda rgb: rgb[1])
    velocity_df["color_b"] = rgb_values.apply(lambda rgb: rgb[2])
    return velocity_df


def build_trunk_visualization_metrics(
    citations: Path,
    articles: Path,
    w_text: float,
    w_references: float,
    w_citers: float,
    min_similarity: float,
    min_size: float,
    max_size: float,
    current_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build trunk subtopics plus size and color metrics."""
    w_text, w_references, w_citers = normalize_weights(
        w_text,
        w_references,
        w_citers,
    )
    graph = build_sap_graph(citations, articles)
    records = extract_trunk_records(graph)
    tfidf_vectors = compute_tfidf_vectors(records)
    reference_sets = compute_trunk_reference_sets(graph, records)
    citer_sets = compute_trunk_citer_sets(graph, records)
    _, _, pairs_df = build_combined_outputs(
        records,
        tfidf_vectors,
        reference_sets,
        citer_sets,
        w_text,
        w_references,
        w_citers,
    )
    _, subtopics_df, summary_df = detect_subtopics(
        records,
        tfidf_vectors,
        pairs_df,
        min_similarity,
    )
    trunk_df = compute_trunk_sizes(graph, subtopics_df, min_size, max_size)
    trunk_df = compute_citation_velocity(trunk_df, current_year)
    trunk_df = trunk_df.sort_values(
        ["trunk_subtopic_id", "is_representative", "sap_rank"],
        ascending=[True, False, False],
    )
    return trunk_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build trunk-only visualization metrics for ToS trunk papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-text", type=float, default=0.40)
    parser.add_argument("--w-references", type=float, default=0.35)
    parser.add_argument("--w-citers", type=float, default=0.25)
    parser.add_argument("--min-similarity", type=float, default=0.18)
    parser.add_argument("--min-size", type=float, default=6.0)
    parser.add_argument("--max-size", type=float, default=25.0)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trunk_df, summary_df = build_trunk_visualization_metrics(
        args.citations,
        args.articles,
        args.w_text,
        args.w_references,
        args.w_citers,
        args.min_similarity,
        args.min_size,
        args.max_size,
        args.current_year,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "trunk_visualization_metrics.csv"
    summary_path = args.output_dir / "trunk_visualization_subtopics.csv"
    trunk_df.to_csv(metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Trunk nodes: {len(trunk_df):,}")
    print(f"Trunk subtopics: {len(summary_df):,}")
    print(
        "Size range: "
        f"{trunk_df['size'].min():.2f} to {trunk_df['size'].max():.2f}"
    )
    print(
        "Citation velocity range: "
        f"{trunk_df['citation_velocity'].min():.4f} to "
        f"{trunk_df['citation_velocity'].max():.4f}"
    )
    print(f"Saved trunk visualization metrics: {metrics_path.resolve()}")
    print(f"Saved trunk visualization subtopics: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
