from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("outputs/graphs/citation_network.gexf")
DEFAULT_OUTPUT_DIR = Path("outputs/leaf_like_nodes")
GEXF_NS = {"g": "http://www.gexf.net/1.2draft"}
YEAR_PATTERN = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export nodes with indegree 0 and outdegree >= 1 from the citation GEXF."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_gexf_structure(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ET.parse(path).getroot()
    attribute_lookup = {
        attr.attrib["id"]: attr.attrib["title"]
        for attr in root.findall(".//g:attributes[@class='node']/g:attribute", GEXF_NS)
    }

    indegree = defaultdict(int)
    outdegree = defaultdict(int)
    edge_rows = []
    for edge in root.findall(".//g:edges/g:edge", GEXF_NS):
        source = edge.attrib["source"]
        target = edge.attrib["target"]
        outdegree[source] += 1
        indegree[target] += 1
        edge_rows.append({"source": source, "target": target})

    node_rows = []
    for node in root.findall(".//g:nodes/g:node", GEXF_NS):
        node_id = node.attrib["id"]
        row = {
            "node_id": node_id,
            "label": node.attrib.get("label", node_id),
            "internal_indegree": indegree[node_id],
            "internal_outdegree": outdegree[node_id],
        }
        for attvalue in node.findall("./g:attvalues/g:attvalue", GEXF_NS):
            key = attribute_lookup.get(attvalue.attrib["for"], attvalue.attrib["for"])
            row[key] = attvalue.attrib.get("value", "")
        node_rows.append(row)

    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def export_leaf_like_nodes(nodes_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    leaf_like_df = nodes_df[
        (nodes_df["internal_indegree"] == 0)
        & (nodes_df["internal_outdegree"] >= 1)
    ].copy()
    leaf_like_df["metadata_year"] = pd.to_numeric(leaf_like_df.get("year"), errors="coerce")
    leaf_like_df["year_from_id"] = pd.to_numeric(
        leaf_like_df["node_id"].astype(str).str.extract(YEAR_PATTERN, expand=False),
        errors="coerce",
    )
    leaf_like_df["effective_year"] = leaf_like_df["metadata_year"].fillna(
        leaf_like_df["year_from_id"]
    )
    leaf_like_df["year_source"] = "metadata"
    leaf_like_df.loc[
        leaf_like_df["metadata_year"].isna() & leaf_like_df["year_from_id"].notna(),
        "year_source",
    ] = "node_id"
    leaf_like_df.loc[leaf_like_df["effective_year"].isna(), "year_source"] = "missing"
    leaf_like_df["year"] = leaf_like_df["effective_year"]
    leaf_like_df["sap_rank"] = pd.to_numeric(leaf_like_df.get("sap_rank"), errors="coerce").fillna(0)
    leaf_like_df = leaf_like_df.sort_values(
        ["effective_year", "internal_outdegree", "sap_rank", "node_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    leaf_like_df.insert(0, "leaf_like_rank", range(1, len(leaf_like_df) + 1))

    preferred_columns = [
        "leaf_like_rank",
        "node_id",
        "title",
        "year",
        "metadata_year",
        "year_from_id",
        "effective_year",
        "year_source",
        "journal",
        "source_title",
        "doi",
        "ToS",
        "sap_rank",
        "internal_indegree",
        "internal_outdegree",
        "has_article_metadata",
        "abstract",
    ]
    existing_columns = [column for column in preferred_columns if column in leaf_like_df.columns]
    remaining_columns = [
        column for column in leaf_like_df.columns if column not in existing_columns
    ]
    leaf_like_df = leaf_like_df[existing_columns + remaining_columns]

    output_dir.mkdir(parents=True, exist_ok=True)
    leaf_like_df.to_csv(output_dir / "indegree_0_outdegree_ge_1_nodes.csv", index=False)
    return leaf_like_df


def export_markdown(leaf_like_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Nodes With Indegree 0 and Outdegree >= 1",
        "",
        f"Total nodes: {len(leaf_like_df)}",
        "",
    ]
    for row in leaf_like_df.itertuples(index=False):
        year = "" if pd.isna(row.effective_year) else str(int(float(row.effective_year)))
        title = str(getattr(row, "title", "") or "No title")
        tos = str(getattr(row, "ToS", "") or "unclassified")
        lines.append(
            f"{int(row.leaf_like_rank)}. **{row.node_id}** ({year}) - {title} "
            f"[ToS={tos}; outdegree={int(row.internal_outdegree)}; year_source={row.year_source}]"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_year_histogram(leaf_like_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    year_counts = (
        leaf_like_df.dropna(subset=["effective_year"])
        .assign(effective_year=lambda df: df["effective_year"].astype(int))
        .groupby("effective_year")
        .size()
        .reset_index(name="node_count")
        .rename(columns={"effective_year": "year"})
        .sort_values("year")
    )
    year_counts.to_csv(output_dir / "indegree_0_outdegree_ge_1_year_counts.csv", index=False)

    year_source_counts = (
        leaf_like_df.groupby("year_source")
        .size()
        .reset_index(name="node_count")
        .sort_values("year_source")
    )
    year_source_counts.to_csv(
        output_dir / "indegree_0_outdegree_ge_1_year_source_counts.csv",
        index=False,
    )

    plt.figure(figsize=(11, 5.8))
    bars = plt.bar(
        year_counts["year"].astype(str),
        year_counts["node_count"],
        color="#7D932D",
        edgecolor="#4F641B",
        linewidth=0.7,
    )
    plt.title("Nodes with indegree 0 and outdegree >= 1 by publication year")
    plt.xlabel("Publication year")
    plt.ylabel("Node count")
    plt.xticks(rotation=45, ha="right")
    plt.bar_label(bars, padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "indegree_0_outdegree_ge_1_year_histogram.png", dpi=200)
    plt.savefig(output_dir / "indegree_0_outdegree_ge_1_year_histogram.svg")
    plt.close()
    return year_counts


def main() -> None:
    args = parse_args()
    nodes_df, edges_df = load_gexf_structure(args.input)
    leaf_like_df = export_leaf_like_nodes(nodes_df, args.output_dir)
    export_markdown(leaf_like_df, args.output_dir / "indegree_0_outdegree_ge_1_nodes.md")
    year_counts = export_year_histogram(leaf_like_df, args.output_dir)

    print(f"Input graph nodes: {len(nodes_df):,}")
    print(f"Input graph edges: {len(edges_df):,}")
    print(f"Nodes with indegree 0 and outdegree >= 1: {len(leaf_like_df):,}")
    print(f"Years with data: {len(year_counts):,}")
    print(
        "Saved node list: "
        f"{(args.output_dir / 'indegree_0_outdegree_ge_1_nodes.csv').resolve()}"
    )
    print(
        "Saved histogram PNG: "
        f"{(args.output_dir / 'indegree_0_outdegree_ge_1_year_histogram.png').resolve()}"
    )


if __name__ == "__main__":
    main()
