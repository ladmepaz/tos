from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd

from fruit_symbol_svg import build_fruit_symbol


DEFAULT_INPUT = Path("outputs/fruits/fruit_candidates_underabsorbed.csv")
DEFAULT_OUTPUT = Path("outputs/fruit_visualization/svg/fruits_top3.svg")
DEFAULT_TOP_OUTPUT = Path("outputs/fruit_visualization/fruits_top3.csv")
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 420
LABEL_COLOR = "#4A413B"
TITLE_COLOR = "#1F130B"
FRUIT_COLOR_LOW = "#E9A195"
FRUIT_COLOR_HIGH = "#A32314"
MIN_FRUIT_RADIUS = 20.0
MAX_FRUIT_RADIUS = 34.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a top-fruits SVG for the enhanced Tree of Science."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-output", type=Path, default=DEFAULT_TOP_OUTPUT)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--background",
        default="none",
        help="SVG background fill. Use 'none' for a transparent asset.",
    )
    return parser.parse_args()


def normalize_series(values: pd.Series) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())
    if values.empty or math.isclose(min_value, max_value):
        return pd.Series(1.0, index=values.index, dtype=float)
    return (values - min_value) / (max_value - min_value)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def interpolate_color(start_hex: str, end_hex: str, score: float) -> str:
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)
    clamped_score = max(0.0, min(1.0, float(score)))
    mixed_rgb = tuple(
        round(start + ((end - start) * clamped_score))
        for start, end in zip(start_rgb, end_rgb)
    )
    return rgb_to_hex(mixed_rgb)


def short_label(node_id: str) -> str:
    parts = [part.strip() for part in str(node_id).split(",")]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return str(node_id)


def select_top_fruits(fruit_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    underabsorbed_df = fruit_df[
        fruit_df["fruit_signal_type"] == "external_fruit_underabsorbed"
    ].copy()
    if len(underabsorbed_df) >= top_n:
        selected_df = underabsorbed_df
    else:
        filler_df = fruit_df[
            fruit_df["fruit_signal_type"].isin(
                ["fruit_candidate", "external_signal_not_matched_in_graph"]
            )
        ].copy()
        selected_df = pd.concat([underabsorbed_df, filler_df], ignore_index=True)

    selected_df = selected_df.sort_values(
        ["fruit_score", "external_citation_velocity", "cited_by"],
        ascending=[False, False, False],
    ).head(top_n)
    return selected_df.reset_index(drop=True)


def add_visual_metrics(fruits_df: pd.DataFrame) -> pd.DataFrame:
    visual_df = fruits_df.copy()
    visual_df["velocity_size_score"] = compute_capped_velocity_score(
        visual_df["external_citation_velocity"]
    )
    visual_df["fruit_color_score"] = normalize_series(visual_df["fruit_score"])
    visual_df["fruit_radius"] = MIN_FRUIT_RADIUS + (
        visual_df["velocity_size_score"] * (MAX_FRUIT_RADIUS - MIN_FRUIT_RADIUS)
    )
    visual_df["fruit_color"] = visual_df["fruit_color_score"].apply(
        lambda score: interpolate_color(FRUIT_COLOR_LOW, FRUIT_COLOR_HIGH, score)
    )
    return visual_df


def compute_capped_velocity_score(values: pd.Series) -> pd.Series:
    """Scale fruit sizes with the visual maximum capped at the second highest velocity."""
    if values.empty:
        return values.astype(float)
    sorted_values = values.sort_values(ascending=False).reset_index(drop=True)
    min_value = float(values.min())
    if len(sorted_values) >= 2:
        max_value = float(sorted_values.iloc[1])
    else:
        max_value = float(sorted_values.iloc[0])

    if math.isclose(min_value, max_value):
        return pd.Series(1.0, index=values.index, dtype=float)
    return ((values - min_value) / (max_value - min_value)).clip(lower=0.0, upper=1.0)


def fruit_positions(member_count: int) -> list[tuple[float, float]]:
    presets = {
        1: [(430.0, 210.0)],
        2: [(285.0, 210.0), (590.0, 210.0)],
        3: [(165.0, 230.0), (430.0, 160.0), (675.0, 245.0)],
    }
    if member_count in presets:
        return presets[member_count]

    positions = []
    for index in range(member_count):
        positions.append((145.0 + (index * 180.0), 210.0))
    return positions


def build_svg(fruits_df: pd.DataFrame, background: str) -> str:
    positions = fruit_positions(len(fruits_df))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="{html.escape(background)}" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        f".fruit-title {{ font-size: 58px; fill: {TITLE_COLOR}; font-weight: 600; }}",
        ".fruit-label { font-size: 19px; font-weight: 600; }",
        ".fruit-metric { font-size: 14px; opacity: 0.78; }",
        "</style>",
        '<text class="fruit-title" x="28" y="72">Fruits</text>',
    ]

    for row, (cx, cy) in zip(fruits_df.itertuples(index=False), positions):
        label = short_label(str(row.SR))
        tooltip = (
            f"{row.title}\n"
            f"External citations: {float(row.cited_by):.0f}\n"
            f"Internal indegree: {float(row.internal_indegree):.0f}\n"
            f"External citation velocity: {float(row.external_citation_velocity):.2f}\n"
            f"Fruit score: {float(row.fruit_score):.3f}"
        )
        radius = float(row.fruit_radius)
        lines.append("<g>")
        lines.append(f"<title>{html.escape(tooltip)}</title>")
        lines.extend(build_fruit_symbol(cx, cy, radius, str(row.fruit_color)))
        label_x = cx + radius + 17.0
        lines.append(
            f'<text class="fruit-label" x="{label_x:.1f}" y="{cy - 4:.1f}">{html.escape(label)}</text>'
        )
        lines.append(
            f'<text class="fruit-metric" x="{label_x:.1f}" y="{cy + 18:.1f}">cited_by {float(row.cited_by):.0f}; indegree {float(row.internal_indegree):.0f}</text>'
        )
        lines.append("</g>")

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fruit_df = pd.read_csv(args.input)
    selected_df = add_visual_metrics(select_top_fruits(fruit_df, args.top_n))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.top_output.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(args.top_output, index=False)
    args.output.write_text(build_svg(selected_df, args.background), encoding="utf-8")

    print(f"Fruits selected: {len(selected_df):,}")
    print(f"Saved fruits CSV: {args.top_output.resolve()}")
    print(f"Saved fruits SVG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
