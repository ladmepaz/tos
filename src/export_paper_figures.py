from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

from fruit_symbol_svg import build_fruit_symbol


DEFAULT_OUTPUT_DIR = Path("outputs/paper_figures")
DEFAULT_MAIN_SVG = Path("outputs/tos_3_visualization_final_3.svg")
DEFAULT_MAIN_JPG = Path("outputs/tos_3_visualization_final_3.jpg")
DEFAULT_LEAF_HISTOGRAM_SVG = Path(
    "outputs/leaf_like_nodes/indegree_0_outdegree_ge_1_year_histogram.svg"
)
DEFAULT_LEAF_HISTOGRAM_PNG = Path(
    "outputs/leaf_like_nodes/indegree_0_outdegree_ge_1_year_histogram.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-ready figure assets for the ToS 3 paper."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--main-svg", type=Path, default=DEFAULT_MAIN_SVG)
    parser.add_argument("--main-jpg", type=Path, default=DEFAULT_MAIN_JPG)
    parser.add_argument("--leaf-histogram-svg", type=Path, default=DEFAULT_LEAF_HISTOGRAM_SVG)
    parser.add_argument("--leaf-histogram-png", type=Path, default=DEFAULT_LEAF_HISTOGRAM_PNG)
    return parser.parse_args()


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        print(f"Skipped missing file: {source}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Saved: {target.resolve()}")


def circle(cx: float, cy: float, r: float, fill: str, label: str) -> str:
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" />\n'
        f'<text class="label" x="{cx + 34}" y="{cy + 6}">{html.escape(label)}</text>'
    )


def leaf_symbol(cx: float, cy: float, scale: float, fill: str, label: str) -> str:
    return (
        f'<g transform="translate({cx} {cy}) rotate(-24) scale({scale})">'
        f'<path d="M 0,-26 C 18,-20 28,-4 21,14 C 14,31 -7,31 -20,16 '
        f'C -32,2 -25,-18 0,-26 Z" fill="{fill}" />'
        '<path d="M -11,15 C -2,3 8,-9 15,-19" stroke="#FFFFFF" '
        'stroke-width="2.3" stroke-linecap="round" fill="none" opacity="0.85" />'
        '<path d="M -4,6 C -12,4 -16,0 -19,-5" stroke="#FFFFFF" '
        'stroke-width="1.4" stroke-linecap="round" fill="none" opacity="0.85" />'
        '<path d="M 3,-2 C 10,1 15,3 19,6" stroke="#FFFFFF" '
        'stroke-width="1.4" stroke-linecap="round" fill="none" opacity="0.85" />'
        '</g>\n'
        f'<text class="label" x="{cx + 34}" y="{cy + 6}">{html.escape(label)}</text>'
    )


def fruit_symbol(cx: float, cy: float, radius: float, fill: str, label: str) -> str:
    symbol = "\n".join(build_fruit_symbol(cx, cy, radius, fill))
    return (
        f"{symbol}\n"
        f'<text class="label" x="{cx + 42}" y="{cy + 6}">{html.escape(label)}</text>'
    )


def write_visual_encoding_legend(output_path: Path) -> None:
    rows = [
        ("Roots", circle(95, 132, 20, "#8D6448", "Foundational papers; size = citations received")),
        ("Trunk", circle(95, 210, 20, "#5C2609", "Consolidating papers; ordered as developmental pathways")),
        ("Branches", circle(95, 288, 17, "#B39A82", "Core papers in recent thematic conversations")),
        ("Leaves", leaf_symbol(95, 366, 0.82, "#D7F22A", "Structural frontier; size = internal references to the tree")),
        ("Dormant Leaves", leaf_symbol(95, 444, 0.82, "#6B452C", "Older frontier papers not absorbed by the local network")),
        ("Fruits", fruit_symbol(95, 526, 20, "#B72217", "External-attention signals; size = external citation velocity")),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="650" viewBox="0 0 1180 650">',
        '<rect width="1180" height="650" fill="#FFFFFF" />',
        '<style>',
        'text { font-family: Georgia, "Times New Roman", serif; fill: #2C2118; }',
        '.title { font-size: 38px; font-weight: 700; }',
        '.subtitle { font-size: 20px; fill: #5D5147; }',
        '.role { font-size: 22px; font-weight: 700; }',
        '.label { font-size: 21px; }',
        '</style>',
        '<text class="title" x="52" y="62">Visual Encoding of Tree of Science 3</text>',
        '<text class="subtitle" x="52" y="96">Analytical role, visual form, and encoded meaning</text>',
    ]
    for index, (role, symbol) in enumerate(rows):
        y = 132 + (index * 78)
        lines.append(f'<text class="role" x="52" y="{y + 6}">{html.escape(role)}</text>')
        lines.append(symbol)
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path.resolve()}")


def write_framework_workflow(output_path: Path) -> None:
    boxes = [
        ("Data", "Scopus + WoS records\\nBibFusion harmonization"),
        ("Citation Network", "Directed local network\\n3,183 nodes; 12,481 edges"),
        ("SAP Structure", "Roots and trunk\\nselective SAP leaves"),
        ("Enhancements", "Root/trunk subtopics\\nbranches, leaves, fruits"),
        ("Visualization", "Role-specific shapes\\nsize, color, and temporal signals"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="430" viewBox="0 0 1500 430">',
        '<rect width="1500" height="430" fill="#FFFFFF" />',
        '<style>',
        'text { font-family: Georgia, "Times New Roman", serif; fill: #2C2118; }',
        '.title { font-size: 38px; font-weight: 700; }',
        '.box-title { font-size: 24px; font-weight: 700; }',
        '.box-text { font-size: 18px; fill: #5D5147; }',
        '</style>',
        '<text class="title" x="48" y="62">Tree of Science 3 Workflow</text>',
    ]
    x0 = 48
    box_w = 245
    gap = 45
    y = 132
    for index, (title, text) in enumerate(boxes):
        x = x0 + index * (box_w + gap)
        lines.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="170" rx="24" '
            'fill="#F6F1EA" stroke="#8D6448" stroke-width="2" />'
        )
        lines.append(f'<text class="box-title" x="{x + 24}" y="{y + 48}">{html.escape(title)}</text>')
        for line_index, line in enumerate(text.split("\\n")):
            lines.append(
                f'<text class="box-text" x="{x + 24}" y="{y + 88 + (line_index * 28)}">'
                f'{html.escape(line)}</text>'
            )
        if index < len(boxes) - 1:
            arrow_x = x + box_w + 10
            arrow_y = y + 85
            lines.append(
                f'<path d="M {arrow_x} {arrow_y} H {arrow_x + gap - 23}" '
                'stroke="#5C2609" stroke-width="4" stroke-linecap="round" />'
            )
            lines.append(
                f'<path d="M {arrow_x + gap - 23} {arrow_y - 10} '
                f'L {arrow_x + gap - 5} {arrow_y} L {arrow_x + gap - 23} {arrow_y + 10} Z" '
                'fill="#5C2609" />'
            )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(args.main_svg, args.output_dir / "figure_1_tos3_visualization.svg")
    copy_if_exists(args.main_jpg, args.output_dir / "figure_1_tos3_visualization.jpg")
    write_visual_encoding_legend(args.output_dir / "figure_2_visual_encoding_legend.svg")
    write_framework_workflow(args.output_dir / "figure_3_tos3_workflow.svg")
    copy_if_exists(
        args.leaf_histogram_svg,
        args.output_dir / "figure_4_leaf_year_histogram.svg",
    )
    copy_if_exists(
        args.leaf_histogram_png,
        args.output_dir / "figure_4_leaf_year_histogram.png",
    )


if __name__ == "__main__":
    main()
