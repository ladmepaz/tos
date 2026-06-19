from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from root_tfidf_similarity import build_sap_graph


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_FRUITS_TOP = Path("outputs/fruit_visualization/fruits_top3.csv")
DEFAULT_METRICS_OUTPUT = Path("outputs/leaf_visualization/leaf_visualization_metrics.csv")
DEFAULT_SVG_OUTPUT = Path("outputs/leaf_visualization/svg/leaves.svg")
DEFAULT_LABELED_SVG_OUTPUT = Path("outputs/leaf_visualization/svg/leaves_labeled.svg")
DEFAULT_LEAF_SYMBOL_OUTPUT = Path("outputs/leaf_visualization/svg/leaf_symbol.svg")
DEFAULT_LEAF_TEMPLATE = Path("outputs/leaf_visualization/svg/leaf_symbol_1.svg")
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 620
LEAF_SYMBOL_CANVAS_WIDTH = 260
LEAF_SYMBOL_CANVAS_HEIGHT = 180
LEAF_COLOR_OLD = "#7D932D"
LEAF_COLOR_NEW = "#BECC2D"
LEAF_VEIN_COLOR = "#4F641B"
LEAF_HIGHLIGHT_COLOR = "#D7E45A"
LABEL_COLOR = "#4A413B"
TITLE_COLOR = "#1F130B"
MIN_LEAF_LENGTH = 19.0
MAX_LEAF_LENGTH = 34.0


@dataclass(frozen=True)
class LeafTemplate:
    path_d: str
    source_transform: str
    viewbox_width: float
    viewbox_height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SAP leaves as green leaf-shaped SVG marks."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--fruits-top", type=Path, default=DEFAULT_FRUITS_TOP)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SVG_OUTPUT)
    parser.add_argument("--labeled-output", type=Path, default=DEFAULT_LABELED_SVG_OUTPUT)
    parser.add_argument("--leaf-symbol-output", type=Path, default=DEFAULT_LEAF_SYMBOL_OUTPUT)
    parser.add_argument(
        "--leaf-template",
        type=Path,
        default=DEFAULT_LEAF_TEMPLATE,
        help="Optional SVG template containing the designer leaf path with id='path566'.",
    )
    parser.add_argument(
        "--include-fruits",
        action="store_true",
        help="Render top fruits as green leaves too. By default, they are marked in the CSV but excluded from the SVG.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Render short paper labels for inspection.",
    )
    parser.add_argument(
        "--background",
        default="none",
        help="SVG background fill. Use 'none' for a transparent asset.",
    )
    return parser.parse_args()


def load_leaf_template(path: Path) -> LeafTemplate | None:
    """Load the designer leaf SVG path, if available."""
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None

    viewbox = root.attrib.get("viewBox", "")
    try:
        _, _, viewbox_width, viewbox_height = [float(part) for part in viewbox.split()]
    except ValueError:
        return None

    path_node = None
    layer_transform = ""
    for element in root.iter():
        if element.attrib.get("id") == "layer1":
            layer_transform = element.attrib.get("transform", "")
        if element.attrib.get("id") == "path566":
            path_node = element
    if path_node is None:
        return None
    path_d = path_node.attrib.get("d", "")
    if not path_d:
        return None
    return LeafTemplate(
        path_d=path_d,
        source_transform=layer_transform,
        viewbox_width=viewbox_width,
        viewbox_height=viewbox_height,
    )


def normalize_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return values.astype(float)
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
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


def load_top_fruit_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    fruit_df = pd.read_csv(path)
    if "SR" not in fruit_df.columns:
        return set()
    return set(fruit_df["SR"].dropna().astype(str))


def extract_leaf_metrics(
    citations: Path,
    articles: Path,
    fruits_top: Path,
) -> pd.DataFrame:
    graph = build_sap_graph(citations, articles)
    fruit_ids = load_top_fruit_ids(fruits_top)
    records = []
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("ToS") != "leaf":
            continue
        records.append(
            {
                "node_id": node_id,
                "year": attrs.get("year"),
                "title": attrs.get("title") or "",
                "journal": attrs.get("journal") or "",
                "source_title": attrs.get("source_title") or "",
                "doi": attrs.get("doi") or "",
                "sap_rank": attrs.get("sap_rank", 0),
                "internal_indegree": graph.in_degree(node_id),
                "internal_outdegree": graph.out_degree(node_id),
                "is_top_fruit": node_id in fruit_ids,
            }
        )

    leaves_df = pd.DataFrame(records)
    leaves_df["year"] = pd.to_numeric(leaves_df["year"], errors="coerce")
    leaves_df["year_score"] = normalize_series(leaves_df["year"].fillna(leaves_df["year"].min()))
    leaves_df["citation_size_score"] = normalize_series(
        leaves_df["internal_indegree"].apply(math.log1p)
    )
    # Leaves should remain visually light; internal citations only create subtle variation.
    leaves_df["leaf_length"] = MIN_LEAF_LENGTH + (
        leaves_df["citation_size_score"] * (MAX_LEAF_LENGTH - MIN_LEAF_LENGTH)
    )
    leaves_df["leaf_width"] = leaves_df["leaf_length"] * 0.43
    leaves_df["leaf_color"] = leaves_df["year_score"].apply(
        lambda score: interpolate_color(LEAF_COLOR_OLD, LEAF_COLOR_NEW, score)
    )
    leaves_df = assign_leaf_layout(leaves_df)
    return leaves_df.sort_values(
        ["is_top_fruit", "year", "sap_rank"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def assign_leaf_layout(leaves_df: pd.DataFrame) -> pd.DataFrame:
    positioned_df = leaves_df.sort_values(
        ["year", "sap_rank"],
        ascending=[False, False],
    ).reset_index(drop=True)
    centers = [
        (230.0, 280.0, 205.0, 120.0),
        (640.0, 195.0, 245.0, 145.0),
        (1035.0, 285.0, 205.0, 125.0),
    ]
    for index in range(len(positioned_df)):
        center_x, center_y, radius_x, radius_y = centers[index % len(centers)]
        ring = index // len(centers)
        angle = math.radians((index * 137.507764) % 360)
        jitter = 0.82 + (0.08 * ((index % 5) - 2))
        positioned_df.loc[index, "x"] = center_x + (
            math.cos(angle) * radius_x * jitter * (0.58 + (0.07 * ring))
        )
        positioned_df.loc[index, "y"] = center_y + (
            math.sin(angle) * radius_y * jitter * (0.58 + (0.06 * ring))
        )
        positioned_df.loc[index, "rotation"] = ((index * 47) % 130) - 65
    return positioned_df


def build_leaf_symbol(
    cx: float,
    cy: float,
    length: float,
    width: float,
    rotation: float,
    fill_color: str,
    template: LeafTemplate | None = None,
) -> list[str]:
    if template is not None:
        viewbox_center_x = template.viewbox_width / 2
        viewbox_center_y = template.viewbox_height / 2
        scale_x = (width * 2.1) / template.viewbox_width
        scale_y = (length * 2.2) / template.viewbox_height
        transform = (
            f"translate({cx:.1f} {cy:.1f}) "
            f"rotate({rotation:.1f}) "
            f"scale({scale_x:.4f} {scale_y:.4f}) "
            f"translate({-viewbox_center_x:.4f} {-viewbox_center_y:.4f})"
        )
        lines = [f'<g transform="{transform}">']
        if template.source_transform:
            lines.append(f'<g transform="{html.escape(template.source_transform)}">')
        lines.append(
            f'<path d="{html.escape(template.path_d)}" fill="{html.escape(fill_color)}" '
            'fill-rule="evenodd" stroke="none" />'
        )
        if template.source_transform:
            lines.append("</g>")
        lines.append("</g>")
        return lines

    vein_width = max(1.0, length * 0.036)
    side_vein_width = max(0.75, vein_width * 0.62)
    vein_color = LEAF_VEIN_COLOR
    highlight_color = LEAF_HIGHLIGHT_COLOR
    return [
        (
            f'<path d="M 0,-1.12 C 0.62,-0.98 1.08,-0.42 0.97,0.14 '
            f'C 0.86,0.72 0.36,1.05 0.02,1.16 '
            f'C -0.38,0.96 -0.92,0.52 -1.03,-0.10 '
            f'C -1.14,-0.72 -0.58,-1.04 0,-1.12 Z" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="{html.escape(fill_color)}" stroke="{html.escape(fill_color)}" stroke-width="0.65" />'
        ),
        (
            f'<path d="M 0,-0.82 C -0.10,-0.38 -0.03,0.25 0.05,0.86" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="none" stroke="{vein_color}" stroke-width="{vein_width:.2f}" stroke-linecap="round" opacity="0.74" />'
        ),
        (
            f'<path d="M -0.02,-0.42 C -0.27,-0.30 -0.47,-0.12 -0.62,0.10" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="none" stroke="{vein_color}" stroke-width="{side_vein_width:.2f}" stroke-linecap="round" opacity="0.60" />'
        ),
        (
            f'<path d="M 0.02,-0.20 C 0.28,-0.06 0.48,0.12 0.62,0.36" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="none" stroke="{vein_color}" stroke-width="{side_vein_width:.2f}" stroke-linecap="round" opacity="0.58" />'
        ),
        (
            f'<path d="M 0.00,0.08 C -0.22,0.22 -0.40,0.42 -0.52,0.66" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="none" stroke="{vein_color}" stroke-width="{side_vein_width:.2f}" stroke-linecap="round" opacity="0.48" />'
        ),
        (
            f'<path d="M -0.42,-0.46 C -0.20,-0.78 0.23,-0.88 0.56,-0.56" '
            f'transform="translate({cx:.1f} {cy:.1f}) rotate({rotation:.1f}) scale({width:.2f} {length:.2f})" '
            f'fill="none" stroke="{highlight_color}" stroke-width="{max(0.8, side_vein_width):.2f}" stroke-linecap="round" opacity="0.28" />'
        ),
    ]


def build_svg(
    leaves_df: pd.DataFrame,
    show_labels: bool,
    background: str,
    template: LeafTemplate | None,
) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">',
        f'<rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="{html.escape(background)}" />',
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        f".leaf-title {{ font-size: 52px; fill: {TITLE_COLOR}; font-weight: 600; }}",
        ".leaf-label { font-size: 13px; font-weight: 600; }",
        "</style>",
        '<g id="leaves">',
    ]
    if show_labels:
        lines.append('<text class="leaf-title" x="24" y="62">Leaves</text>')

    for row in leaves_df.itertuples(index=False):
        tooltip = (
            f"{row.title or row.node_id}\n"
            f"Year: {float(row.year):.0f}\n"
            f"Internal citations: {int(row.internal_indegree)}\n"
            f"SAP rank: {float(row.sap_rank):.0f}"
        )
        lines.append("<g>")
        lines.append(f"<title>{html.escape(tooltip)}</title>")
        lines.extend(
            build_leaf_symbol(
                float(row.x),
                float(row.y),
                float(row.leaf_length),
                float(row.leaf_width),
                float(row.rotation),
                str(row.leaf_color),
                template,
            )
        )
        if show_labels:
            lines.append(
                f'<text class="leaf-label" x="{float(row.x) + float(row.leaf_width) + 8:.1f}" y="{float(row.y) + 4:.1f}">{html.escape(short_label(str(row.node_id)))}</text>'
            )
        lines.append("</g>")

    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def build_leaf_symbol_svg(background: str, template: LeafTemplate | None) -> str:
    cx = LEAF_SYMBOL_CANVAS_WIDTH / 2
    cy = LEAF_SYMBOL_CANVAS_HEIGHT / 2
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{LEAF_SYMBOL_CANVAS_WIDTH}" height="{LEAF_SYMBOL_CANVAS_HEIGHT}" viewBox="0 0 {LEAF_SYMBOL_CANVAS_WIDTH} {LEAF_SYMBOL_CANVAS_HEIGHT}">',
        f'<rect width="{LEAF_SYMBOL_CANVAS_WIDTH}" height="{LEAF_SYMBOL_CANVAS_HEIGHT}" fill="{html.escape(background)}" />',
        '<g id="tos-leaf-symbol">',
    ]
    lines.extend(
        build_leaf_symbol(
            cx,
            cy,
            length=62.0,
            width=28.0,
            rotation=58.0,
            fill_color="#AFC42D",
            template=template,
        )
    )
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    template = load_leaf_template(args.leaf_template)
    leaves_df = extract_leaf_metrics(args.citations, args.articles, args.fruits_top)
    svg_df = leaves_df.copy() if args.include_fruits else leaves_df[~leaves_df["is_top_fruit"]].copy()

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.labeled_output.parent.mkdir(parents=True, exist_ok=True)
    args.leaf_symbol_output.parent.mkdir(parents=True, exist_ok=True)
    leaves_df.to_csv(args.metrics_output, index=False)
    args.output.write_text(
        build_svg(
            svg_df,
            show_labels=False,
            background=args.background,
            template=template,
        ),
        encoding="utf-8",
    )
    args.labeled_output.write_text(
        build_svg(
            svg_df,
            show_labels=True,
            background=args.background,
            template=template,
        ),
        encoding="utf-8",
    )
    args.leaf_symbol_output.write_text(
        build_leaf_symbol_svg(background=args.background, template=template),
        encoding="utf-8",
    )

    print(f"SAP leaves: {len(leaves_df):,}")
    print(f"Top fruits marked: {int(leaves_df['is_top_fruit'].sum()):,}")
    print(f"Leaves rendered: {len(svg_df):,}")
    print(f"Saved leaf metrics: {args.metrics_output.resolve()}")
    print(f"Saved leaf SVG: {args.output.resolve()}")
    print(f"Saved labeled leaf SVG: {args.labeled_output.resolve()}")
    print(f"Saved leaf symbol SVG: {args.leaf_symbol_output.resolve()}")
    print(
        "Leaf template: "
        f"{args.leaf_template.resolve() if template is not None else 'procedural fallback'}"
    )


if __name__ == "__main__":
    main()
