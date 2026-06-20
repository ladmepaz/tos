from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import math
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

import pandas as pd

from export_leaf_visualization_svg import (
    DEFAULT_ARTICLES,
    DEFAULT_CITATIONS,
    DEFAULT_FRUITS_TOP,
    DEFAULT_LEAF_TEMPLATE,
    build_leaf_symbol,
    extract_leaf_metrics,
    load_leaf_template,
)


DEFAULT_SHAPE = Path("outputs/leaf_visualization/svg/leaves_shape.svg")
DEFAULT_OUTPUT = Path("outputs/leaf_visualization/svg/leaves_canopy_shape.svg")
DEFAULT_DEBUG_OUTPUT = Path("outputs/leaf_visualization/svg/leaves_canopy_shape_debug.svg")
DEFAULT_METRICS_OUTPUT = Path("outputs/leaf_visualization/leaves_canopy_shape_metrics.csv")
LABEL_COLOR = "#4A413B"
TITLE_COLOR = "#1F130B"
MIN_LEAF_LENGTH = 12.0
MAX_LEAF_LENGTH = 21.0
CANVAS_PADDING = 34.0


@dataclass(frozen=True)
class CanopyShape:
    path_d: str
    source_transform: str
    viewbox_width: float
    viewbox_height: float
    points: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place SAP leaves inside a designer canopy silhouette."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--fruits-top", type=Path, default=DEFAULT_FRUITS_TOP)
    parser.add_argument("--shape", type=Path, default=DEFAULT_SHAPE)
    parser.add_argument("--leaf-template", type=Path, default=DEFAULT_LEAF_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug-output", type=Path, default=DEFAULT_DEBUG_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-fruits",
        action="store_true",
        help="Render top fruits as leaves too. By default they remain excluded from the leaf layer.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Render short labels for placement inspection.",
    )
    parser.add_argument(
        "--clip-to-shape",
        action="store_true",
        help="Clip leaves to the canopy silhouette. Disabled by default so leaves can extend naturally beyond the outline.",
    )
    parser.add_argument(
        "--background",
        default="none",
        help="SVG background fill. Use 'none' for a transparent asset.",
    )
    return parser.parse_args()


def parse_translate(transform: str) -> tuple[float, float]:
    match = re.search(
        r"translate\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"(?:[\s,]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*\)",
        transform,
    )
    if not match:
        return 0.0, 0.0
    tx = float(match.group(1))
    ty = float(match.group(2) or 0.0)
    return tx, ty


def load_canopy_shape(path: Path) -> CanopyShape:
    if not path.exists():
        raise FileNotFoundError(f"Canopy shape SVG not found: {path}")

    root = ET.parse(path).getroot()
    viewbox = root.attrib.get("viewBox", "")
    try:
        _, _, viewbox_width, viewbox_height = [float(part) for part in viewbox.split()]
    except ValueError as exc:
        raise ValueError(f"Shape SVG must define a numeric viewBox: {path}") from exc

    black_path = None
    source_transform = ""
    for element in root.iter():
        if element.attrib.get("id") == "layer1":
            source_transform = element.attrib.get("transform", "")
        style = element.attrib.get("style", "").replace(" ", "").lower()
        fill = element.attrib.get("fill", "").lower()
        if element.tag.endswith("path") and (
            element.attrib.get("id") == "path144"
            or "fill:#000000" in style
            or fill in {"#000000", "black"}
        ):
            black_path = element
            break

    if black_path is None:
        raise ValueError(f"Could not find a black canopy path in: {path}")
    path_d = black_path.attrib.get("d", "")
    if not path_d:
        raise ValueError(f"Canopy path has no 'd' attribute: {path}")

    tx, ty = parse_translate(source_transform)
    points = svg_path_to_points(path_d, tx=tx, ty=ty)
    if len(points) < 3:
        raise ValueError(f"Canopy path could not be converted to polygon points: {path}")

    return CanopyShape(
        path_d=path_d,
        source_transform=source_transform,
        viewbox_width=viewbox_width,
        viewbox_height=viewbox_height,
        points=points,
    )


def tokenize_svg_path(path_d: str) -> list[str]:
    return re.findall(
        r"[MmZzLlHhVvCcSsQqTt]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        path_d,
    )


def is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def read_float(tokens: list[str], index: int) -> tuple[float, int]:
    return float(tokens[index]), index + 1


def cubic_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    inv = 1.0 - t
    x = (
        (inv**3 * p0[0])
        + (3 * inv**2 * t * p1[0])
        + (3 * inv * t**2 * p2[0])
        + (t**3 * p3[0])
    )
    y = (
        (inv**3 * p0[1])
        + (3 * inv**2 * t * p1[1])
        + (3 * inv * t**2 * p2[1])
        + (t**3 * p3[1])
    )
    return x, y


def svg_path_to_points(path_d: str, tx: float, ty: float) -> list[tuple[float, float]]:
    """Approximate the SVG path as polygon points for placement tests."""
    tokens = tokenize_svg_path(path_d)
    points: list[tuple[float, float]] = []
    index = 0
    command = ""
    x = 0.0
    y = 0.0
    start_x = 0.0
    start_y = 0.0

    while index < len(tokens):
        if is_command(tokens[index]):
            command = tokens[index]
            index += 1
        if not command:
            break

        relative = command.islower()
        op = command.upper()

        if op == "M":
            first_pair = True
            while index < len(tokens) and not is_command(tokens[index]):
                nx, index = read_float(tokens, index)
                ny, index = read_float(tokens, index)
                if relative:
                    nx += x
                    ny += y
                x, y = nx, ny
                if first_pair:
                    start_x, start_y = x, y
                    first_pair = False
                points.append((x + tx, y + ty))
            command = "l" if relative else "L"
        elif op == "L":
            while index < len(tokens) and not is_command(tokens[index]):
                nx, index = read_float(tokens, index)
                ny, index = read_float(tokens, index)
                if relative:
                    nx += x
                    ny += y
                x, y = nx, ny
                points.append((x + tx, y + ty))
        elif op == "H":
            while index < len(tokens) and not is_command(tokens[index]):
                nx, index = read_float(tokens, index)
                x = x + nx if relative else nx
                points.append((x + tx, y + ty))
        elif op == "V":
            while index < len(tokens) and not is_command(tokens[index]):
                ny, index = read_float(tokens, index)
                y = y + ny if relative else ny
                points.append((x + tx, y + ty))
        elif op == "C":
            while index < len(tokens) and not is_command(tokens[index]):
                x1, index = read_float(tokens, index)
                y1, index = read_float(tokens, index)
                x2, index = read_float(tokens, index)
                y2, index = read_float(tokens, index)
                x3, index = read_float(tokens, index)
                y3, index = read_float(tokens, index)
                if relative:
                    p1 = (x + x1, y + y1)
                    p2 = (x + x2, y + y2)
                    p3 = (x + x3, y + y3)
                else:
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    p3 = (x3, y3)
                p0 = (x, y)
                for step in range(1, 9):
                    px, py = cubic_point(p0, p1, p2, p3, step / 8)
                    points.append((px + tx, py + ty))
                x, y = p3
        elif op == "Z":
            x, y = start_x, start_y
            points.append((x + tx, y + ty))
            command = ""
        else:
            raise ValueError(f"Unsupported SVG path command in canopy shape: {command}")

    return points


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = (current_y > y) != (previous_y > y)
        if intersects:
            x_intersect = (
                (previous_x - current_x) * (y - current_y) / (previous_y - current_y)
            ) + current_x
            if x < x_intersect:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def polygon_bounds(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def layout_leaves_in_canopy(
    leaves_df: pd.DataFrame,
    shape: CanopyShape,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    positioned_df = leaves_df.sort_values(
        ["year", "sap_rank"],
        ascending=[False, False],
    ).reset_index(drop=True)
    if positioned_df.empty:
        return positioned_df

    min_x, min_y, max_x, max_y = polygon_bounds(shape.points)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    placed: list[tuple[float, float, float]] = []

    for index, row in positioned_df.iterrows():
        leaf_score = float(row.get("citation_size_score", 0.0))
        length = MIN_LEAF_LENGTH + (leaf_score * (MAX_LEAF_LENGTH - MIN_LEAF_LENGTH))
        width = length * 0.559
        min_distance = length * 1.18
        accepted = None

        for attempt in range(4000):
            candidate_x = rng.uniform(min_x, max_x)
            candidate_y = rng.uniform(min_y, max_y)
            if not point_in_polygon(candidate_x, candidate_y, shape.points):
                continue
            distance_factor = 1.0 if attempt < 1800 else 0.82 if attempt < 3000 else 0.68
            if all(
                math.hypot(candidate_x - other_x, candidate_y - other_y)
                >= ((min_distance + other_radius) * 0.5 * distance_factor)
                for other_x, other_y, other_radius in placed
            ):
                accepted = (candidate_x, candidate_y)
                break

        if accepted is None:
            # Fallback keeps the SVG complete even when the silhouette is crowded.
            for _ in range(12000):
                candidate_x = rng.uniform(min_x, max_x)
                candidate_y = rng.uniform(min_y, max_y)
                if point_in_polygon(candidate_x, candidate_y, shape.points):
                    accepted = (candidate_x, candidate_y)
                    break

        if accepted is None:
            raise RuntimeError("Could not place a leaf inside the canopy shape.")

        x, y = accepted
        radial_angle = math.degrees(math.atan2(y - center_y, x - center_x))
        rotation = radial_angle + rng.uniform(-42.0, 42.0)
        positioned_df.loc[index, "x"] = x
        positioned_df.loc[index, "y"] = y
        positioned_df.loc[index, "rotation"] = rotation
        positioned_df.loc[index, "leaf_length"] = length
        positioned_df.loc[index, "leaf_width"] = width
        placed.append((x, y, min_distance))

    return positioned_df


def short_label(node_id: str) -> str:
    parts = [part.strip() for part in str(node_id).split(",")]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return str(node_id)


def canopy_clip_path(shape: CanopyShape) -> list[str]:
    lines = [
        '<clipPath id="canopy-clip" clipPathUnits="userSpaceOnUse">',
    ]
    if shape.source_transform:
        lines.append(f'<g transform="{html.escape(shape.source_transform)}">')
    lines.append(
        f'<path d="{html.escape(shape.path_d)}" fill="#000000" fill-rule="evenodd" />'
    )
    if shape.source_transform:
        lines.append("</g>")
    lines.append("</clipPath>")
    return lines


def canopy_debug_path(shape: CanopyShape) -> list[str]:
    lines = []
    if shape.source_transform:
        lines.append(f'<g transform="{html.escape(shape.source_transform)}">')
    lines.append(
        f'<path d="{html.escape(shape.path_d)}" fill="#000000" fill-opacity="0.08" '
        'stroke="#111111" stroke-opacity="0.25" stroke-width="0.7" />'
    )
    if shape.source_transform:
        lines.append("</g>")
    return lines


def build_svg(
    leaves_df: pd.DataFrame,
    shape: CanopyShape,
    template,
    background: str,
    show_labels: bool,
    show_shape: bool,
    clip_to_shape: bool,
) -> str:
    canvas_width = shape.viewbox_width + (CANVAS_PADDING * 2)
    canvas_height = shape.viewbox_height + (CANVAS_PADDING * 2)
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:.6f}mm" '
            f'height="{canvas_height:.6f}mm" viewBox="0 0 {canvas_width:.6f} {canvas_height:.6f}">'
        ),
        f'<rect width="{canvas_width:.6f}" height="{canvas_height:.6f}" fill="{html.escape(background)}" />',
        "<defs>",
        *canopy_clip_path(shape),
        "</defs>",
        "<style>",
        f'text {{ font-family: Georgia, "Times New Roman", serif; fill: {LABEL_COLOR}; }}',
        f".leaf-title {{ font-size: 18px; fill: {TITLE_COLOR}; font-weight: 600; }}",
        ".leaf-label { font-size: 4.5px; font-weight: 600; }",
        "</style>",
        f'<g id="canopy-layer" transform="translate({CANVAS_PADDING:.1f} {CANVAS_PADDING:.1f})">',
    ]
    if show_shape:
        lines.append('<g id="canopy-debug-shape">')
        lines.extend(canopy_debug_path(shape))
        lines.append("</g>")
    if show_labels:
        lines.append('<text class="leaf-title" x="10" y="24">Leaves</text>')

    if clip_to_shape:
        lines.append('<g id="canopy-leaves" clip-path="url(#canopy-clip)">')
    else:
        lines.append('<g id="canopy-leaves">')
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
                f'<text class="leaf-label" x="{float(row.x) + float(row.leaf_width) + 1.6:.1f}" '
                f'y="{float(row.y) + 1.4:.1f}">{html.escape(short_label(str(row.node_id)))}</text>'
            )
        lines.append("</g>")
    lines.append("</g>")
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    shape = load_canopy_shape(args.shape)
    template = load_leaf_template(args.leaf_template)
    leaves_df = extract_leaf_metrics(args.citations, args.articles, args.fruits_top)
    leaves_df = leaves_df.copy() if args.include_fruits else leaves_df[~leaves_df["is_top_fruit"]].copy()
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

    print(f"Leaves rendered inside canopy: {len(leaves_df):,}")
    print(f"Clip to canopy shape: {args.clip_to_shape}")
    print(f"Saved canopy leaf SVG: {args.output.resolve()}")
    print(f"Saved canopy debug SVG: {args.debug_output.resolve()}")
    print(f"Saved canopy metrics: {args.metrics_output.resolve()}")


if __name__ == "__main__":
    main()
