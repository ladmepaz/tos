from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/branch_visualization/branch_visualization_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/branch_visualization/svg/branch_visualization_direct.svg")
CANVAS_WIDTH = 1360
CANVAS_HEIGHT = 820
BACKGROUND_COLOR = "#F7F2EA"
LABEL_COLOR = "#4A413B"
PX_PER_MM = 96.0 / 25.4
MIN_DIAMETER_MM = 6.0
MAX_DIAMETER_MM = 25.0
BRANCH_LAYOUT = {
    "branch_1": {
        "x": 85.0,
        "y": 585.0,
        "dx": 70.0,
        "dy": -52.0,
        "side": 1,
    },
    "branch_2": {
        "x": 455.0,
        "y": 590.0,
        "dx": 66.0,
        "dy": -49.0,
        "side": -1,
    },
    "branch_3": {
        "x": 830.0,
        "y": 585.0,
        "dx": 70.0,
        "dy": -52.0,
        "side": 1,
    },
}
ROLE_PRIORITY = {
    "core": 1,
    "peripheral": 2,
    "background_methodological": 3,
    "missing_metadata": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export deterministic branch-only SVGs with ToS branch roles."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Render paper labels next to branch circles.",
    )
    return parser.parse_args()


def short_label(node_id: str) -> str:
    parts = [part.strip() for part in str(node_id).split(",")]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return str(node_id)


def role_priority(role: str) -> int:
    return ROLE_PRIORITY.get(str(role), 5)


def size_to_radius_px(size_value: float, min_size: float, max_size: float) -> float:
    if math.isclose(min_size, max_size):
        diameter_mm = (MIN_DIAMETER_MM + MAX_DIAMETER_MM) / 2.0
    else:
        normalized = (float(size_value) - min_size) / (max_size - min_size)
        diameter_mm = MIN_DIAMETER_MM + (normalized * (MAX_DIAMETER_MM - MIN_DIAMETER_MM))
    return (diameter_mm * PX_PER_MM) / 2.0


def node_sort_key(row: pd.Series) -> tuple[int, float, float, float, str]:
    return (
        role_priority(str(row["branch_member_role"])),
        -float(row.get("branch_core_score", 0) or 0),
        -float(row.get("year", 0) or 0),
        -float(row.get("citations_received", 0) or 0),
        str(row["node_id"]),
    )


def place_core_nodes(group_df: pd.DataFrame, layout: dict[str, float]) -> pd.DataFrame:
    core_df = group_df[group_df["branch_member_role"] == "core"].copy()
    core_df = core_df.sort_values(
        ["year", "branch_core_score", "citations_received"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    for index in range(len(core_df)):
        core_df.loc[index, "x"] = layout["x"] + (index * layout["dx"])
        core_df.loc[index, "y"] = layout["y"] + (index * layout["dy"])
    return core_df


def place_side_nodes(
    group_df: pd.DataFrame,
    layout: dict[str, float],
    role: str,
    start_index: int,
    distance: float,
) -> pd.DataFrame:
    side_df = group_df[group_df["branch_member_role"] == role].copy()
    side_df = side_df.sort_values(
        ["branch_core_score", "year", "citations_received"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    side_sign = float(layout["side"])
    for index in range(len(side_df)):
        anchor_index = start_index + index
        alternating = 1.0 if index % 2 == 0 else -0.65
        side_df.loc[index, "x"] = layout["x"] + (anchor_index * layout["dx"]) + 34.0
        side_df.loc[index, "y"] = (
            layout["y"]
            + (anchor_index * layout["dy"])
            + (side_sign * alternating * distance)
        )
    return side_df


def compute_layout(nodes_df: pd.DataFrame) -> pd.DataFrame:
    positioned_groups = []
    for branch_label in sorted(nodes_df["branch_label"].unique()):
        if branch_label not in BRANCH_LAYOUT:
            continue
        layout = BRANCH_LAYOUT[branch_label]
        group_df = nodes_df[nodes_df["branch_label"] == branch_label].copy()
        group_df["_sort_key"] = group_df.apply(node_sort_key, axis=1)
        group_df = group_df.sort_values("_sort_key").drop(columns=["_sort_key"])

        core_df = place_core_nodes(group_df, layout)
        peripheral_df = place_side_nodes(
            group_df,
            layout,
            "peripheral",
            start_index=1,
            distance=95.0,
        )
        background_df = place_side_nodes(
            group_df,
            layout,
            "background_methodological",
            start_index=2,
            distance=155.0,
        )
        missing_df = place_side_nodes(
            group_df,
            layout,
            "missing_metadata",
            start_index=3,
            distance=185.0,
        )
        positioned_groups.append(
            pd.concat(
                [core_df, peripheral_df, background_df, missing_df],
                ignore_index=True,
            )
        )

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
        ".node-label { font-size: 15px; font-weight: 600; }",
        ".branch-label { font-size: 18px; font-weight: 600; letter-spacing: 0.4px; }",
        "</style>",
    ]

    for branch_label in sorted(layout_df["branch_label"].unique()):
        branch_df = layout_df[layout_df["branch_label"] == branch_label].copy()
        lines.append(f'<g id="{html.escape(branch_label)}">')
        label_x = float(branch_df["x"].min())
        label_y = float(branch_df["y"].max()) + 76.0
        lines.append(
            f'<text class="branch-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(branch_label.replace("_", " ").title())}</text>'
        )

        for row in branch_df.itertuples(index=False):
            radius = size_to_radius_px(float(row.size), min_size, max_size)
            fill_color = html.escape(str(row.color))
            label = short_label(str(row.node_id))
            title = str(row.title) if isinstance(row.title, str) and row.title else str(row.node_id)
            tooltip = (
                f"{title}\n"
                f"Branch: {row.branch_label}\n"
                f"Role: {row.branch_member_role}\n"
                f"Citations: {int(float(row.citations_received))}\n"
                f"Citation velocity: {float(row.citation_velocity):.2f}\n"
                f"Core score: {float(row.branch_core_score):.3f}"
            )
            lines.append("<g>")
            lines.append(f"<title>{html.escape(tooltip)}</title>")
            lines.append(
                f'<circle cx="{float(row.x):.1f}" cy="{float(row.y):.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="3" />'
            )
            if show_labels:
                label_x = float(row.x) + radius + 11.0
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
    print(f"Nodes: {len(nodes_df):,}")
    print(f"Saved SVG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
