from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("outputs/branch_visualization/branch_visualization_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/branch_visualization/svg/branch_visualization_direct.svg")
DEFAULT_SPLIT_OUTPUT_DIR = Path("outputs/branch_visualization/svg/branches")
DEFAULT_CORE_OUTPUT_DIR = Path("outputs/branch_visualization/svg/core_papers")
DEFAULT_PERIPHERAL_OUTPUT_DIR = Path("outputs/branch_visualization/svg/peripheral_papers")
DEFAULT_BACKGROUND_OUTPUT_DIR = Path("outputs/branch_visualization/svg/background_methodological")
DEFAULT_MISSING_OUTPUT_DIR = Path("outputs/branch_visualization/svg/missing_metadata")
DEFAULT_OVERVIEW_OUTPUT = Path("outputs/branch_visualization/svg/branch_visualization_overview.svg")
CANVAS_WIDTH = 1360
CANVAS_HEIGHT = 820
CARD_CANVAS_WIDTH = 760
CARD_CANVAS_HEIGHT = 620
OVERVIEW_CANVAS_WIDTH = 2450
OVERVIEW_ROW_HEIGHT = 430
OVERVIEW_TOP_MARGIN = 26
OVERVIEW_BOTTOM_MARGIN = 55
BACKGROUND_COLOR = "#F7F2EA"
CARD_BACKGROUND_COLOR = "#FFFFFF"
CARD_BORDER_COLOR = "#CFCFCF"
LABEL_COLOR = "#4A413B"
CARD_TITLE_COLOR = "#1F130B"
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
OVERVIEW_SECTIONS = [
    {
        "role": "core",
        "title": "Core papers",
        "x": 0.0,
        "width": 615.0,
    },
    {
        "role": "peripheral",
        "title": "Peripheral papers",
        "x": 620.0,
        "width": 520.0,
    },
    {
        "role": "background_methodological",
        "title": "Background/methodological",
        "x": 1165.0,
        "width": 500.0,
    },
    {
        "role": "missing_metadata",
        "title": "Missing metadata",
        "x": 1695.0,
        "width": 650.0,
    },
]
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
    parser.add_argument(
        "--split-branches",
        action="store_true",
        help="Also export one tight-canvas SVG per branch.",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=DEFAULT_SPLIT_OUTPUT_DIR,
        help="Directory for per-branch SVG files.",
    )
    parser.add_argument(
        "--export-core-cards",
        action="store_true",
        help="Export one card-style SVG per branch containing only core papers.",
    )
    parser.add_argument(
        "--core-output-dir",
        type=Path,
        default=DEFAULT_CORE_OUTPUT_DIR,
        help="Directory for core-paper branch card SVG files.",
    )
    parser.add_argument(
        "--export-peripheral-cards",
        action="store_true",
        help="Export one card-style SVG per branch containing only peripheral papers.",
    )
    parser.add_argument(
        "--peripheral-output-dir",
        type=Path,
        default=DEFAULT_PERIPHERAL_OUTPUT_DIR,
        help="Directory for peripheral-paper branch card SVG files.",
    )
    parser.add_argument(
        "--export-background-cards",
        action="store_true",
        help="Export one card-style SVG per branch containing only background/methodological papers.",
    )
    parser.add_argument(
        "--background-output-dir",
        type=Path,
        default=DEFAULT_BACKGROUND_OUTPUT_DIR,
        help="Directory for background/methodological branch card SVG files.",
    )
    parser.add_argument(
        "--export-missing-cards",
        action="store_true",
        help="Export one card-style SVG per branch containing only missing-metadata papers.",
    )
    parser.add_argument(
        "--missing-output-dir",
        type=Path,
        default=DEFAULT_MISSING_OUTPUT_DIR,
        help="Directory for missing-metadata branch card SVG files.",
    )
    parser.add_argument(
        "--export-overview",
        action="store_true",
        help="Export a large combined SVG with all branches and all role sections.",
    )
    parser.add_argument(
        "--overview-output",
        type=Path,
        default=DEFAULT_OVERVIEW_OUTPUT,
        help="Output SVG path for the combined branch overview.",
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


def node_tooltip(row) -> str:
    title = str(row.title) if isinstance(row.title, str) and row.title else str(row.node_id)
    return (
        f"{title}\n"
        f"Branch: {row.branch_label}\n"
        f"Role: {row.branch_member_role}\n"
        f"Citations: {int(float(row.citations_received))}\n"
        f"Citation velocity: {float(row.citation_velocity):.2f}\n"
        f"Core score: {float(row.branch_core_score):.3f}"
    )


def render_nodes(
    layout_df: pd.DataFrame,
    min_size: float,
    max_size: float,
    show_labels: bool,
    include_branch_titles: bool,
) -> list[str]:
    lines: list[str] = []
    for branch_label in sorted(layout_df["branch_label"].unique()):
        branch_df = layout_df[layout_df["branch_label"] == branch_label].copy()
        lines.append(f'<g id="{html.escape(branch_label)}">')
        if include_branch_titles:
            label_x = float(branch_df["x"].min())
            label_y = float(branch_df["y"].max()) + 76.0
            lines.append(
                f'<text class="branch-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(branch_label.replace("_", " ").title())}</text>'
            )

        for row in branch_df.itertuples(index=False):
            radius = size_to_radius_px(float(row.size), min_size, max_size)
            fill_color = html.escape(str(row.color))
            lines.append("<g>")
            lines.append(f"<title>{html.escape(node_tooltip(row))}</title>")
            lines.append(
                f'<circle cx="{float(row.x):.1f}" cy="{float(row.y):.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="3" />'
            )
            if show_labels:
                label = short_label(str(row.node_id))
                label_x = float(row.x) + radius + 11.0
                label_y = float(row.y) + 5.0
                lines.append(
                    f'<text class="node-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(label)}</text>'
                )
            lines.append("</g>")

        lines.append("</g>")
    return lines


def svg_header(width: float, height: float, background_color: str) -> list[str]:
    rect_fill = background_color if background_color != "none" else "none"
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}">',
        f'<rect width="{width:.1f}" height="{height:.1f}" fill="{rect_fill}" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        ".node-label { font-size: 15px; font-weight: 600; }",
        ".branch-label { font-size: 18px; font-weight: 600; letter-spacing: 0.4px; }",
        f".core-title {{ font-size: 58px; fill: {CARD_TITLE_COLOR}; }}",
        f".core-footer {{ font-size: 58px; fill: {CARD_TITLE_COLOR}; }}",
        ".core-label { font-size: 19px; font-weight: 600; }",
        "</style>",
    ]
    return lines


def build_svg(
    layout_df: pd.DataFrame,
    min_size: float,
    max_size: float,
    show_labels: bool,
    include_branch_titles: bool = True,
) -> str:
    lines = svg_header(CANVAS_WIDTH, CANVAS_HEIGHT, BACKGROUND_COLOR)
    lines.extend(
        render_nodes(
            layout_df,
            min_size,
            max_size,
            show_labels,
            include_branch_titles,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


def branch_bounds(
    branch_df: pd.DataFrame,
    min_size: float,
    max_size: float,
    show_labels: bool,
) -> tuple[float, float, float, float]:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for row in branch_df.itertuples(index=False):
        radius = size_to_radius_px(float(row.size), min_size, max_size)
        min_x = min(min_x, float(row.x) - radius - 8.0)
        min_y = min(min_y, float(row.y) - radius - 8.0)
        max_x = max(max_x, float(row.x) + radius + 8.0)
        max_y = max(max_y, float(row.y) + radius + 8.0)
        if show_labels:
            label = short_label(str(row.node_id))
            max_x = max(max_x, float(row.x) + radius + 20.0 + (len(label) * 9.0))
    return min_x, min_y, max_x, max_y


def build_branch_svg(
    branch_df: pd.DataFrame,
    min_size: float,
    max_size: float,
    show_labels: bool,
) -> str:
    padding = 28.0
    min_x, min_y, max_x, max_y = branch_bounds(
        branch_df,
        min_size,
        max_size,
        show_labels,
    )
    width = max_x - min_x + (padding * 2)
    height = max_y - min_y + (padding * 2)
    translated_df = branch_df.copy()
    translated_df["x"] = translated_df["x"] - min_x + padding
    translated_df["y"] = translated_df["y"] - min_y + padding

    lines = svg_header(width, height, "none")
    lines.extend(
        render_nodes(
            translated_df,
            min_size,
            max_size,
            show_labels,
            include_branch_titles=False,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines)


def export_split_branches(
    layout_df: pd.DataFrame,
    split_output_dir: Path,
    min_size: float,
    max_size: float,
    show_labels: bool,
) -> list[Path]:
    split_output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = []
    label_suffix = "_labeled" if show_labels else ""
    for branch_label in sorted(layout_df["branch_label"].unique()):
        branch_df = layout_df[layout_df["branch_label"] == branch_label].copy()
        output_path = split_output_dir / f"{branch_label}{label_suffix}.svg"
        output_path.write_text(
            build_branch_svg(branch_df, min_size, max_size, show_labels),
            encoding="utf-8",
        )
        exported_paths.append(output_path)
    return exported_paths


def paper_card_positions(member_count: int) -> list[tuple[float, float]]:
    presets: dict[int, list[tuple[float, float]]] = {
        1: [(320.0, 290.0)],
        2: [(210.0, 260.0), (455.0, 360.0)],
        3: [(260.0, 165.0), (120.0, 350.0), (445.0, 355.0)],
        4: [(250.0, 160.0), (105.0, 335.0), (430.0, 305.0), (250.0, 485.0)],
        5: [(250.0, 135.0), (105.0, 245.0), (430.0, 290.0), (200.0, 500.0), (430.0, 440.0)],
        6: [(230.0, 135.0), (100.0, 245.0), (50.0, 375.0), (200.0, 500.0), (430.0, 430.0), (450.0, 285.0)],
    }
    if member_count in presets:
        return presets[member_count]

    center_x = CARD_CANVAS_WIDTH / 2.0
    center_y = CARD_CANVAS_HEIGHT / 2.0
    radius_x = 235.0
    radius_y = 175.0
    return [
        (
            center_x + (radius_x * math.cos((2 * math.pi * index) / member_count)),
            center_y + (radius_y * math.sin((2 * math.pi * index) / member_count)),
        )
        for index in range(member_count)
    ]


def build_paper_card_svg(
    branch_df: pd.DataFrame,
    member_role: str,
    footer_label: str,
    min_size: float,
    max_size: float,
) -> str:
    card_df = branch_df[branch_df["branch_member_role"] == member_role].copy()
    card_df = card_df.sort_values(
        ["branch_rank", "year", "branch_core_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    positions = paper_card_positions(len(card_df))
    branch_label = str(branch_df["branch_label"].iloc[0]).replace("_", " ").title()

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_CANVAS_WIDTH}" height="{CARD_CANVAS_HEIGHT}" viewBox="0 0 {CARD_CANVAS_WIDTH} {CARD_CANVAS_HEIGHT}">',
        f'<rect x="1" y="1" width="{CARD_CANVAS_WIDTH - 2}" height="{CARD_CANVAS_HEIGHT - 2}" fill="{CARD_BACKGROUND_COLOR}" stroke="{CARD_BORDER_COLOR}" stroke-width="2" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        f".core-title {{ font-size: 58px; fill: {CARD_TITLE_COLOR}; }}",
        f".core-footer {{ font-size: 58px; fill: {CARD_TITLE_COLOR}; }}",
        ".core-label { font-size: 19px; font-weight: 600; }",
        "</style>",
        f'<text class="core-title" x="18" y="70">{html.escape(branch_label)}</text>',
    ]

    if card_df.empty:
        lines.append(
            '<text class="core-label" x="80" y="300">No papers in this category</text>'
        )

    for row, (cx, cy) in zip(card_df.itertuples(index=False), positions):
        radius = size_to_radius_px(float(row.size), min_size, max_size)
        fill_color = html.escape(str(row.color))
        label = short_label(str(row.node_id))
        label_x = cx + radius + 13.0
        label_y = cy + 6.0
        lines.append("<g>")
        lines.append(f"<title>{html.escape(node_tooltip(row))}</title>")
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="3" />'
        )
        lines.append(
            f'<text class="core-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(label)}</text>'
        )
        lines.append("</g>")

    lines.append(
        f'<text class="core-footer" x="48" y="{CARD_CANVAS_HEIGHT - 25}">{html.escape(footer_label)}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def export_role_cards(
    layout_df: pd.DataFrame,
    output_dir: Path,
    member_role: str,
    footer_label: str,
    filename_suffix: str,
    min_size: float,
    max_size: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_paths = []
    for branch_label in sorted(layout_df["branch_label"].unique()):
        branch_df = layout_df[layout_df["branch_label"] == branch_label].copy()
        output_path = output_dir / f"{branch_label}_{filename_suffix}.svg"
        output_path.write_text(
            build_paper_card_svg(
                branch_df,
                member_role,
                footer_label,
                min_size,
                max_size,
            ),
            encoding="utf-8",
        )
        exported_paths.append(output_path)
    return exported_paths


def overview_positions(member_count: int, section_width: float) -> list[tuple[float, float]]:
    presets: dict[int, list[tuple[float, float]]] = {
        0: [],
        1: [(section_width * 0.50, 205.0)],
        2: [(section_width * 0.36, 150.0), (section_width * 0.58, 260.0)],
        3: [(section_width * 0.25, 165.0), (section_width * 0.55, 210.0), (section_width * 0.40, 315.0)],
        4: [(section_width * 0.21, 145.0), (section_width * 0.52, 170.0), (section_width * 0.38, 300.0), (section_width * 0.72, 260.0)],
        5: [(section_width * 0.20, 128.0), (section_width * 0.48, 160.0), (section_width * 0.76, 182.0), (section_width * 0.38, 292.0), (section_width * 0.66, 315.0)],
        6: [(section_width * 0.24, 115.0), (section_width * 0.09, 205.0), (section_width * 0.45, 185.0), (section_width * 0.08, 305.0), (section_width * 0.35, 335.0), (section_width * 0.62, 275.0)],
    }
    if member_count in presets:
        return presets[member_count]

    positions = []
    columns = 3
    for index in range(member_count):
        column = index % columns
        row = index // columns
        positions.append(
            (
                section_width * (0.18 + (column * 0.27)),
                120.0 + (row * 95.0),
            )
        )
    return positions


def render_overview_section(
    branch_df: pd.DataFrame,
    section: dict[str, float | str],
    row_y: float,
    min_size: float,
    max_size: float,
) -> list[str]:
    section_df = branch_df[
        branch_df["branch_member_role"] == str(section["role"])
    ].copy()
    section_df = section_df.sort_values(
        ["branch_rank", "year", "branch_core_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    positions = overview_positions(len(section_df), float(section["width"]))
    section_x = float(section["x"])
    lines: list[str] = []

    if section_df.empty:
        lines.append(
            f'<text class="overview-empty" x="{section_x + 80:.1f}" y="{row_y + 205:.1f}">No papers</text>'
        )

    for row, (offset_x, offset_y) in zip(section_df.itertuples(index=False), positions):
        cx = section_x + offset_x
        cy = row_y + offset_y
        radius = size_to_radius_px(float(row.size), min_size, max_size)
        fill_color = html.escape(str(row.color))
        label = short_label(str(row.node_id))
        label_x = cx + radius + 12.0
        label_y = cy + 6.0
        lines.append("<g>")
        lines.append(f"<title>{html.escape(node_tooltip(row))}</title>")
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fill_color}" stroke="{fill_color}" stroke-width="3" />'
        )
        lines.append(
            f'<text class="overview-node-label" x="{label_x:.1f}" y="{label_y:.1f}">{html.escape(label)}</text>'
        )
        lines.append("</g>")

    title_y = row_y + OVERVIEW_ROW_HEIGHT - 30.0
    lines.append(
        f'<text class="overview-section-title" x="{section_x + 20:.1f}" y="{title_y:.1f}">{html.escape(str(section["title"]))}</text>'
    )
    return lines


def build_overview_svg(
    layout_df: pd.DataFrame,
    min_size: float,
    max_size: float,
) -> str:
    branch_labels = sorted(layout_df["branch_label"].unique())
    height = (
        OVERVIEW_TOP_MARGIN
        + (len(branch_labels) * OVERVIEW_ROW_HEIGHT)
        + OVERVIEW_BOTTOM_MARGIN
    )
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{OVERVIEW_CANVAS_WIDTH}" height="{height:.1f}" viewBox="0 0 {OVERVIEW_CANVAS_WIDTH} {height:.1f}">',
        '<rect width="100%" height="100%" fill="#FFFFFF" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        f".overview-branch-title {{ font-size: 54px; fill: {CARD_TITLE_COLOR}; font-weight: 600; }}",
        f".overview-section-title {{ font-size: 48px; fill: {CARD_TITLE_COLOR}; font-weight: 600; }}",
        ".overview-node-label { font-size: 18px; font-weight: 600; }",
        ".overview-empty { font-size: 18px; font-style: italic; opacity: 0.7; }",
        "</style>",
    ]

    for index, branch_label in enumerate(branch_labels):
        row_y = OVERVIEW_TOP_MARGIN + (index * OVERVIEW_ROW_HEIGHT)
        branch_df = layout_df[layout_df["branch_label"] == branch_label].copy()
        lines.append(
            f'<text class="overview-branch-title" x="0" y="{row_y + 50:.1f}">{html.escape(branch_label.replace("_", " ").title())}</text>'
        )
        for section in OVERVIEW_SECTIONS:
            lines.extend(
                render_overview_section(
                    branch_df,
                    section,
                    row_y,
                    min_size,
                    max_size,
                )
            )

    lines.append("</svg>")
    return "\n".join(lines)


def export_overview(
    layout_df: pd.DataFrame,
    output_path: Path,
    min_size: float,
    max_size: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_overview_svg(layout_df, min_size, max_size),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.input)
    layout_df = compute_layout(nodes_df)
    min_size = float(layout_df["size"].min())
    max_size = float(layout_df["size"].max())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg = build_svg(layout_df, min_size, max_size, args.show_labels)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Nodes: {len(nodes_df):,}")
    print(f"Saved SVG: {args.output.resolve()}")
    if args.split_branches:
        exported_paths = export_split_branches(
            layout_df,
            args.split_output_dir,
            min_size,
            max_size,
            args.show_labels,
        )
        for path in exported_paths:
            print(f"Saved branch SVG: {path.resolve()}")
    if args.export_core_cards:
        exported_paths = export_role_cards(
            layout_df,
            args.core_output_dir,
            "core",
            "Core papers",
            "core_papers",
            min_size,
            max_size,
        )
        for path in exported_paths:
            print(f"Saved core-card SVG: {path.resolve()}")
    if args.export_peripheral_cards:
        exported_paths = export_role_cards(
            layout_df,
            args.peripheral_output_dir,
            "peripheral",
            "Peripheral papers",
            "peripheral_papers",
            min_size,
            max_size,
        )
        for path in exported_paths:
            print(f"Saved peripheral-card SVG: {path.resolve()}")
    if args.export_background_cards:
        exported_paths = export_role_cards(
            layout_df,
            args.background_output_dir,
            "background_methodological",
            "Background papers",
            "background_methodological",
            min_size,
            max_size,
        )
        for path in exported_paths:
            print(f"Saved background-card SVG: {path.resolve()}")
    if args.export_missing_cards:
        exported_paths = export_role_cards(
            layout_df,
            args.missing_output_dir,
            "missing_metadata",
            "Missing metadata",
            "missing_metadata",
            min_size,
            max_size,
        )
        for path in exported_paths:
            print(f"Saved missing-metadata-card SVG: {path.resolve()}")
    if args.export_overview:
        output_path = export_overview(
            layout_df,
            args.overview_output,
            min_size,
            max_size,
        )
        print(f"Saved branch overview SVG: {output_path.resolve()}")


if __name__ == "__main__":
    main()
