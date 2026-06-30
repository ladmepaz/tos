from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from export_canopy_leaf_visualization_svg import (
    build_svg,
    layout_leaves_in_canopy,
    load_canopy_shape,
)
from export_leaf_visualization_svg import (
    DEFAULT_LEAF_TEMPLATE,
    interpolate_color,
    load_leaf_template,
    normalize_series,
)


DEFAULT_INPUT = Path("outputs/leaf_like_nodes/indegree_0_outdegree_ge_1_nodes.csv")
DEFAULT_SHAPE = Path("outputs/leaf_visualization/svg/leaves_shape.svg")
DEFAULT_OUTPUT = Path("outputs/leaf_visualization/svg/leaves_canopy_shape_121.svg")
DEFAULT_DEBUG_OUTPUT = Path("outputs/leaf_visualization/svg/leaves_canopy_shape_121_debug.svg")
DEFAULT_METRICS_OUTPUT = Path("outputs/leaf_visualization/leaves_canopy_shape_121_metrics.csv")
RECENT_WINDOW_YEARS = 5
DORMANT_OLD_LEAF_COLOR = "#5B3A24"
DORMANT_LEAF_COLOR = "#8A5F3E"
DARK_GREEN_LEAF_COLOR = "#244F1C"
ACTIVE_LEAF_COLOR = "#D7F22A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all indegree-0/outdegree>=1 leaf-like nodes in the canopy."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--shape", type=Path, default=DEFAULT_SHAPE)
    parser.add_argument("--leaf-template", type=Path, default=DEFAULT_LEAF_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug-output", type=Path, default=DEFAULT_DEBUG_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--seed", type=int, default=121)
    parser.add_argument(
        "--background",
        default="none",
        help="SVG background fill. Use 'none' for a transparent asset.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Render short labels for inspection.",
    )
    parser.add_argument(
        "--clip-to-shape",
        action="store_true",
        help="Clip leaves to the canopy silhouette. Disabled by default.",
    )
    return parser.parse_args()


def assign_leaf_temporal_colors(df: pd.DataFrame) -> pd.DataFrame:
    """Color leaves by age: bright green when newest, dark green by year five, brown after."""
    colored_df = df.copy()
    newest_year = float(colored_df["year"].max())
    oldest_year = float(colored_df["year"].min())
    dormant_year_span = max(1.0, newest_year - RECENT_WINDOW_YEARS - oldest_year)

    def temporal_color(year: float) -> tuple[float, str, str]:
        if pd.isna(year):
            return float("nan"), DORMANT_OLD_LEAF_COLOR, "unknown_year_leaf"

        leaf_age = newest_year - float(year)
        if leaf_age <= RECENT_WINDOW_YEARS:
            freshness_score = 1.0 - (leaf_age / RECENT_WINDOW_YEARS)
            color = interpolate_color(
                DARK_GREEN_LEAF_COLOR,
                ACTIVE_LEAF_COLOR,
                freshness_score,
            )
            temporal_class = (
                "fresh_frontier_leaf"
                if leaf_age == 0
                else "recent_dark_green_leaf"
            )
            return leaf_age, color, temporal_class

        dormant_score = max(
            0.0,
            min(
                1.0,
                (float(year) - oldest_year) / dormant_year_span,
            ),
        )
        color = interpolate_color(
            DORMANT_OLD_LEAF_COLOR,
            DORMANT_LEAF_COLOR,
            dormant_score,
        )
        return leaf_age, color, "dormant_leaf"

    temporal_values = colored_df["year"].apply(temporal_color)
    colored_df["leaf_age"] = temporal_values.apply(lambda value: value[0])
    colored_df["leaf_color"] = temporal_values.apply(lambda value: value[1])
    colored_df["leaf_temporal_class"] = temporal_values.apply(lambda value: value[2])
    return colored_df


def prepare_leaf_like_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    year_col = "effective_year" if "effective_year" in df.columns else "year"
    df["year"] = pd.to_numeric(df[year_col], errors="coerce")
    df["sap_rank"] = pd.to_numeric(df.get("sap_rank"), errors="coerce").fillna(0)
    df["internal_indegree"] = pd.to_numeric(
        df.get("internal_indegree"),
        errors="coerce",
    ).fillna(0)
    df["internal_outdegree"] = pd.to_numeric(
        df.get("internal_outdegree"),
        errors="coerce",
    ).fillna(0)
    df["title"] = df.get("title", "").fillna("")
    df["node_id"] = df["node_id"].astype(str)

    df["citation_size_score"] = normalize_series(df["internal_outdegree"].clip(lower=0))
    df = assign_leaf_temporal_colors(df)
    df["leaf_type"] = df["ToS"].fillna("").apply(
        lambda value: "sap_leaf" if str(value) == "leaf" else "dormant_or_frontier_leaf"
    )
    return df.sort_values(
        ["year", "internal_outdegree", "sap_rank", "node_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    leaves_df = prepare_leaf_like_metrics(args.input)
    shape = load_canopy_shape(args.shape)
    template = load_leaf_template(args.leaf_template)
    leaves_df = layout_leaves_in_canopy(leaves_df, shape, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.debug_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    leaves_df.to_csv(args.metrics_output, index=False)
    args.output.write_text(
        build_svg(
            leaves_df,
            shape=shape,
            template=template,
            background=args.background,
            show_labels=args.show_labels,
            show_shape=False,
            clip_to_shape=args.clip_to_shape,
        ),
        encoding="utf-8",
    )
    args.debug_output.write_text(
        build_svg(
            leaves_df,
            shape=shape,
            template=template,
            background=args.background,
            show_labels=args.show_labels,
            show_shape=True,
            clip_to_shape=args.clip_to_shape,
        ),
        encoding="utf-8",
    )

    print(f"Leaf-like nodes rendered: {len(leaves_df):,}")
    print(f"SAP leaves in rendered set: {(leaves_df['leaf_type'] == 'sap_leaf').sum():,}")
    print(
        "Dormant/frontier non-SAP leaves: "
        f"{(leaves_df['leaf_type'] != 'sap_leaf').sum():,}"
    )
    print(f"Saved 121-leaf canopy SVG: {args.output.resolve()}")
    print(f"Saved debug SVG: {args.debug_output.resolve()}")
    print(f"Saved metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()
