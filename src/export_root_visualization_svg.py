from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/root_visualization/root_visualization_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/root_visualization/svg/root_visualization_direct.svg")
CANVAS_WIDTH = 1300
CANVAS_HEIGHT = 820
BACKGROUND_COLOR = "#F7F2EA"
LABEL_COLOR = "#4A413B"
PX_PER_MM = 96.0 / 25.4
MIN_DIAMETER_MM = 6.0
MAX_DIAMETER_MM = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic root-only SVG with fixed sizes and colors."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_cluster_centers(cluster_count: int) -> list[tuple[float, float]]:
    if cluster_count <= 1:
        return [(CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2)]
    if cluster_count == 2:
        return [(380.0, 430.0), (920.0, 430.0)]
    return [(250.0, 470.0), (650.0, 290.0), (1050.0, 470.0)]


def build_cluster_slots(member_count: int) -> list[tuple[float, float]]:
    presets: dict[int, list[tuple[float, float]]] = {
        1: [(0.0, 0.0)],
        2: [(-80.0, 0.0), (80.0, 0.0)],
        3: [(0.0, -10.0), (-95.0, 95.0), (95.0, 95.0)],
        4: [(0.0, -20.0), (-115.0, 75.0), (115.0, 75.0), (0.0, 165.0)],
        5: [(0.0, -25.0), (-120.0, 65.0), (120.0, 65.0), (-82.0, 175.0), (82.0, 175.0)],
        6: [(0.0, -35.0), (-125.0, 40.0), (125.0, 40.0), (-140.0, 165.0), (0.0, 195.0), (140.0, 165.0)],
    }
    if member_count in presets:
        return presets[member_count]

    slots: list[tuple[float, float]] = [(0.0, 0.0)]
    outer_count = member_count - 1
    for index in range(outer_count):
        angle = math.radians(210 - ((index + 1) * 150 / max(1, outer_count - 1)))
        slots.append((165.0 * math.cos(angle), 125.0 + (135.0 * math.sin(angle))))
    return slots


def short_label(node_id: str) -> str:
    parts = [part.strip() for part in str(node_id).split(",")]
    if len(parts) >= 2:
        author = parts[0]
        year = parts[1]
        return f"{author} {year}"
    return str(node_id)


def cluster_caption(row: pd.Series) -> str:
    return f"Cluster {int(row['cluster_order'])}"


def size_to_radius_px(size_value: float, min_size: float, max_size: float) -> float:
    if math.isclose(min_size, max_size):
        diameter_mm = (MIN_DIAMETER_MM + MAX_DIAMETER_MM) / 2.0
    else:
        normalized = (float(size_value) - min_size) / (max_size - min_size)
        diameter_mm = MIN_DIAMETER_MM + (normalized * (MAX_DIAMETER_MM - MIN_DIAMETER_MM))
    return (diameter_mm * PX_PER_MM) / 2.0


def build_svg(nodes_df: pd.DataFrame) -> str:
    ordered_clusters = (
        nodes_df[["subtopic_id", "cluster_order", "subtopic_label"]]
        .drop_duplicates()
        .sort_values("cluster_order")
        .reset_index(drop=True)
    )
    min_size = float(nodes_df["size"].min())
    max_size = float(nodes_df["size"].max())
    centers = build_cluster_centers(len(ordered_clusters))
    center_map = {
        row.subtopic_id: centers[index]
        for index, row in enumerate(ordered_clusters.itertuples(index=False))
    }

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="{BACKGROUND_COLOR}" />',
        '<style>',
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        '.cluster-title { font-size: 20px; font-weight: 600; letter-spacing: 0.5px; }',
        '.node-label { font-size: 18px; }',
        '</style>',
    ]

    for cluster_row in ordered_clusters.itertuples(index=False):
        cluster_nodes = nodes_df[nodes_df["subtopic_id"] == cluster_row.subtopic_id].copy()
        cluster_nodes = cluster_nodes.sort_values(
            ["is_representative", "root_anchor_score", "citations_received"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        slots = build_cluster_slots(len(cluster_nodes))
        center_x, center_y = center_map[cluster_row.subtopic_id]

        lines.append(f'<g id="{html.escape(cluster_row.subtopic_id)}">')
        lines.append(
            f'<text class="cluster-title" x="{center_x:.1f}" y="{center_y - 135:.1f}" text-anchor="middle">{html.escape(cluster_caption(pd.Series(cluster_row._asdict())))}</text>'
        )

        for (_, node_row), (offset_x, offset_y) in zip(cluster_nodes.iterrows(), slots):
            cx = center_x + offset_x
            cy = center_y + offset_y
            radius = size_to_radius_px(float(node_row["size"]), min_size, max_size)
            label = short_label(str(node_row["node_id"]))
            title = str(node_row["title"]) if isinstance(node_row["title"], str) else str(node_row["node_id"])
            tooltip = (
                f"{title}\n"
                f"Subtopic: {node_row['subtopic_id']}\n"
                f"Citations: {int(node_row['citations_received'])}\n"
                f"Citation velocity: {float(node_row['citation_velocity']):.2f}\n"
                f"Size level: {float(node_row['size']):.1f}"
            )
            lines.append("<g>")
            lines.append(f"<title>{html.escape(tooltip)}</title>")
            fill_color = html.escape(str(node_row["color"]))
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="6" />'
            )
            lines.append(
                f'<text class="node-label" x="{cx:.1f}" y="{cy + radius + 28:.1f}" text-anchor="middle">{html.escape(label)}</text>'
            )
            lines.append("</g>")

        lines.append("</g>")

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(nodes_df)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Nodes: {len(nodes_df):,}")
    print(f"Saved SVG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
