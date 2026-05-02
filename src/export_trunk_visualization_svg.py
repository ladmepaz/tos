from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/trunk_visualization/trunk_visualization_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/trunk_visualization/svg/trunk_visualization_direct.svg")
CANVAS_WIDTH = 1180
CANVAS_HEIGHT = 760
BACKGROUND_COLOR = "#F7F2EA"
LABEL_COLOR = "#4A413B"
PX_PER_MM = 96.0 / 25.4
MIN_DIAMETER_MM = 6.0
MAX_DIAMETER_MM = 25.0
MAIN_SUBTOPIC_ORDER = [
    "trunk_subtopic_2",
    "trunk_subtopic_1",
    "trunk_subtopic_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic trunk-only SVG with fixed sizes and colors."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Render paper labels next to trunk circles.",
    )
    return parser.parse_args()


def short_label(node_id: str) -> str:
    parts = [part.strip() for part in str(node_id).split(",")]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return str(node_id)


def size_to_radius_px(size_value: float, min_size: float, max_size: float) -> float:
    if math.isclose(min_size, max_size):
        diameter_mm = (MIN_DIAMETER_MM + MAX_DIAMETER_MM) / 2.0
    else:
        normalized = (float(size_value) - min_size) / (max_size - min_size)
        diameter_mm = MIN_DIAMETER_MM + (normalized * (MAX_DIAMETER_MM - MIN_DIAMETER_MM))
    return (diameter_mm * PX_PER_MM) / 2.0


def compute_layout(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Build a three-column trunk layout for the main subtopics."""
    layout_df = nodes_df[
        nodes_df["trunk_subtopic_id"].isin(MAIN_SUBTOPIC_ORDER)
    ].copy()
    layout_df["publication_year"] = pd.to_numeric(
        layout_df["year"],
        errors="coerce",
    ).fillna(0)
    layout_df["subtopic_order"] = layout_df["trunk_subtopic_id"].map(
        {subtopic_id: index for index, subtopic_id in enumerate(MAIN_SUBTOPIC_ORDER)}
    )

    column_x = {
        "trunk_subtopic_2": 95.0,
        "trunk_subtopic_1": 455.0,
        "trunk_subtopic_3": 850.0,
    }
    top_y = {
        "trunk_subtopic_2": 72.0,
        "trunk_subtopic_1": 42.0,
        "trunk_subtopic_3": 205.0,
    }
    row_gap = {
        "trunk_subtopic_2": 120.0,
        "trunk_subtopic_1": 104.0,
        "trunk_subtopic_3": 220.0,
    }

    positioned_groups = []
    for subtopic_id in MAIN_SUBTOPIC_ORDER:
        group_df = layout_df[layout_df["trunk_subtopic_id"] == subtopic_id].copy()
        group_df = group_df.sort_values(
            ["publication_year", "sap_rank", "citations_received"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        for index in range(len(group_df)):
            group_df.loc[index, "x"] = column_x[subtopic_id]
            group_df.loc[index, "y"] = top_y[subtopic_id] + (index * row_gap[subtopic_id])
        positioned_groups.append(group_df)

    return pd.concat(positioned_groups, ignore_index=True)


def build_svg(nodes_df: pd.DataFrame, show_labels: bool) -> str:
    layout_df = compute_layout(nodes_df)
    min_size = float(layout_df["size"].min())
    max_size = float(layout_df["size"].max())

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="{BACKGROUND_COLOR}" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        ".node-label { font-size: 16px; font-weight: 600; }",
        "</style>",
        '<g id="trunk">',
    ]

    for row in layout_df.itertuples(index=False):
        radius = size_to_radius_px(float(row.size), min_size, max_size)
        fill_color = html.escape(str(row.color))
        label = short_label(str(row.node_id))
        title = str(row.title) if isinstance(row.title, str) and row.title else str(row.node_id)
        tooltip = (
            f"{title}\n"
            f"Subtopic: {row.trunk_subtopic_id}\n"
            f"Citations: {int(float(row.citations_received))}\n"
            f"Citation velocity: {float(row.citation_velocity):.2f}\n"
            f"Size: {float(row.size):.1f}"
        )
        lines.append("<g>")
        lines.append(f"<title>{html.escape(tooltip)}</title>")
        lines.append(
            f'<circle cx="{float(row.x):.1f}" cy="{float(row.y):.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="3" />'
        )
        if show_labels:
            label_x = float(row.x) + radius + 13.0
            label_y = float(row.y) + 5.0
            lines.append(
                f'<text class="node-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(label)}</text>'
            )
        lines.append("</g>")

    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(nodes_df, args.show_labels)
    args.output.write_text(svg, encoding="utf-8")
    displayed_nodes = nodes_df[
        nodes_df["trunk_subtopic_id"].isin(MAIN_SUBTOPIC_ORDER)
    ]
    print(f"Nodes: {len(displayed_nodes):,}")
    print(f"Saved SVG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
