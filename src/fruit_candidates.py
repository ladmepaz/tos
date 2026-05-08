from __future__ import annotations

import argparse
from collections import deque
from datetime import date
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from root_tfidf_similarity import (
    build_sap_graph,
    compute_tfidf_vectors,
    cosine_similarity,
    tokenize,
)
from trunk_visualization_metrics import normalize_series


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/fruits")
FIELD_TOS_PREFIXES = ("branch_",)
FIELD_TOS_LABELS = {"root", "trunk"}
EXCLUDED_FRUIT_TOS_LABELS = {"root", "trunk"}


def read_article_source(
    articles: Path,
    user_data_template: Path | None,
    user_data_sheet: str,
) -> pd.DataFrame:
    """Read the source containing external cited_by counts."""
    if user_data_template and user_data_template.exists():
        return pd.read_excel(user_data_template, sheet_name=user_data_sheet)
    return pd.read_csv(articles)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected BibFusion/user-template column names."""
    normalized_df = df.copy()
    column_lookup = {column.lower().strip(): column for column in normalized_df.columns}
    rename_map = {}
    for target in ["SR", "title", "abstract", "year", "journal", "source_title", "doi", "cited_by"]:
        source = column_lookup.get(target.lower())
        if source and source != target:
            rename_map[source] = target
    if rename_map:
        normalized_df = normalized_df.rename(columns=rename_map)
    return normalized_df


def boolean_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def filter_main_articles(df: pd.DataFrame, include_non_main: bool) -> pd.DataFrame:
    """Prefer the user/main article corpus over cited-reference-only rows."""
    if include_non_main:
        return df.copy()

    main_columns = [
        column
        for column in ["__is_main", "ismainarticle", "ismain_wos", "ismain_scopus"]
        if column in df.columns
    ]
    if not main_columns:
        return df.copy()

    mask = pd.Series(False, index=df.index)
    for column in main_columns:
        mask = mask | boolean_series(df[column])
    return df[mask].copy()


def graph_tos_label(graph: nx.DiGraph, node_id: str) -> str:
    if node_id not in graph:
        return ""
    return str(graph.nodes[node_id].get("ToS", ""))


def is_excluded_tos_label(tos_label: str) -> bool:
    return tos_label in EXCLUDED_FRUIT_TOS_LABELS or tos_label.startswith(FIELD_TOS_PREFIXES)


def is_field_node(attrs: dict) -> bool:
    tos_label = str(attrs.get("ToS", ""))
    return tos_label in FIELD_TOS_LABELS or tos_label.startswith(FIELD_TOS_PREFIXES)


def compute_field_distances(graph: nx.DiGraph, cutoff: int) -> dict[str, int]:
    """Compute undirected distance from every reachable node to roots/trunk/branches."""
    field_nodes = [
        node
        for node, attrs in graph.nodes(data=True)
        if is_field_node(attrs)
    ]
    if not field_nodes:
        return {}
    undirected = graph.to_undirected()
    distances = {node: 0 for node in field_nodes}
    queue = deque(field_nodes)
    while queue:
        current_node = queue.popleft()
        current_distance = distances[current_node]
        if current_distance >= cutoff:
            continue
        for neighbor in undirected.neighbors(current_node):
            if neighbor in distances:
                continue
            distances[neighbor] = current_distance + 1
            queue.append(neighbor)
    return distances


def direct_field_neighbor_count(graph: nx.DiGraph, node_id: str) -> int:
    if node_id not in graph:
        return 0
    neighbors = set(graph.predecessors(node_id)) | set(graph.successors(node_id))
    return sum(1 for neighbor in neighbors if is_field_node(graph.nodes[neighbor]))


def build_anchor_records(graph: nx.DiGraph) -> list[dict]:
    records = []
    for node_id, attrs in graph.nodes(data=True):
        if not is_field_node(attrs):
            continue
        text = f"{attrs.get('title') or ''} {attrs.get('abstract') or ''}".strip()
        tokens = tokenize(text)
        if tokens:
            records.append({"root_id": f"anchor::{node_id}", "tokens": tokens})
    return records


def compute_text_relevance(
    graph: nx.DiGraph,
    candidates_df: pd.DataFrame,
) -> pd.Series:
    """Estimate candidate textual proximity to root/trunk/branch papers."""
    anchor_records = build_anchor_records(graph)
    candidate_records = []
    for row in candidates_df.itertuples(index=False):
        text = f"{getattr(row, 'title', '') or ''} {getattr(row, 'abstract', '') or ''}"
        candidate_records.append(
            {
                "root_id": f"candidate::{row.SR}",
                "node_id": row.SR,
                "tokens": tokenize(text),
            }
        )

    all_records = anchor_records + candidate_records
    if not anchor_records or not candidate_records:
        return pd.Series(0.0, index=candidates_df.index)

    vectors = compute_tfidf_vectors(all_records)
    anchor_ids = [record["root_id"] for record in anchor_records]
    scores = {}
    for record in candidate_records:
        candidate_vector = vectors.get(record["root_id"], {})
        if not candidate_vector:
            scores[record["node_id"]] = 0.0
            continue
        scores[record["node_id"]] = max(
            cosine_similarity(candidate_vector, vectors.get(anchor_id, {}))
            for anchor_id in anchor_ids
        )
    return candidates_df["SR"].map(scores).fillna(0.0)


def add_fruit_scores(
    graph: nx.DiGraph,
    articles_df: pd.DataFrame,
    current_year: int,
    recent_years: int,
    include_non_main: bool,
    field_distance_cutoff: int,
    w_external_velocity: float,
    w_attention_gap: float,
    w_network_proximity: float,
    w_text_relevance: float,
) -> pd.DataFrame:
    """Build the fruit candidate table."""
    source_df = normalize_columns(articles_df)
    required_columns = {"SR", "year", "cited_by"}
    missing_columns = required_columns - set(source_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    source_df = filter_main_articles(source_df, include_non_main)
    source_df = source_df.dropna(subset=["SR"]).copy()
    source_df["SR"] = source_df["SR"].astype(str).str.strip()
    source_df["year"] = pd.to_numeric(source_df["year"], errors="coerce")
    source_df["cited_by"] = pd.to_numeric(source_df["cited_by"], errors="coerce").fillna(0)
    source_df = source_df[source_df["year"].notna()].copy()
    cutoff_year = current_year - recent_years
    source_df = source_df[source_df["year"] >= cutoff_year].copy()
    source_df = source_df.sort_values("cited_by", ascending=False)
    source_df = source_df.drop_duplicates(subset=["SR"], keep="first").copy()

    source_df["baseline_ToS"] = source_df["SR"].map(lambda node_id: graph_tos_label(graph, node_id))
    source_df = source_df[
        ~source_df["baseline_ToS"].map(is_excluded_tos_label)
    ].copy()
    source_df["in_current_graph"] = source_df["SR"].map(lambda node_id: node_id in graph)
    source_df["internal_indegree"] = source_df["SR"].map(
        lambda node_id: graph.in_degree(node_id) if node_id in graph else 0
    ).astype(float)
    source_df["years_since_publication"] = (
        current_year - source_df["year"] + 1
    ).clip(lower=1)
    source_df["external_citation_velocity"] = (
        source_df["cited_by"] / source_df["years_since_publication"]
    )

    source_df["external_citation_score"] = normalize_series(
        source_df["cited_by"].apply(math.log1p)
    )
    source_df["internal_citation_score"] = normalize_series(
        source_df["internal_indegree"].apply(math.log1p)
    )
    source_df["external_velocity_score"] = normalize_series(
        source_df["external_citation_velocity"].apply(math.log1p)
    )
    source_df["external_attention_gap_raw"] = (
        source_df["external_citation_score"] - source_df["internal_citation_score"]
    ).clip(lower=0)
    source_df["external_attention_gap_score"] = normalize_series(
        source_df["external_attention_gap_raw"]
    )

    distances = compute_field_distances(graph, field_distance_cutoff)
    source_df["field_distance"] = source_df["SR"].map(distances)
    source_df["field_distance_score"] = source_df["field_distance"].map(
        lambda value: 1 / (1 + value) if pd.notna(value) else 0.0
    )
    source_df["direct_field_neighbor_count"] = source_df["SR"].map(
        lambda node_id: direct_field_neighbor_count(graph, node_id)
    )
    source_df["direct_field_neighbor_score"] = normalize_series(
        source_df["direct_field_neighbor_count"].apply(math.log1p)
    )
    source_df["network_proximity_score"] = (
        0.65 * source_df["field_distance_score"]
        + 0.35 * source_df["direct_field_neighbor_score"]
    )

    source_df["text_relevance_raw"] = compute_text_relevance(graph, source_df)
    source_df["text_relevance_score"] = normalize_series(source_df["text_relevance_raw"])

    weight_sum = (
        w_external_velocity
        + w_attention_gap
        + w_network_proximity
        + w_text_relevance
    )
    source_df["fruit_score"] = (
        (w_external_velocity / weight_sum) * source_df["external_velocity_score"]
        + (w_attention_gap / weight_sum) * source_df["external_attention_gap_score"]
        + (w_network_proximity / weight_sum) * source_df["network_proximity_score"]
        + (w_text_relevance / weight_sum) * source_df["text_relevance_score"]
    )
    source_df["fruit_signal_type"] = source_df.apply(classify_fruit_signal, axis=1)

    output_columns = [
        "SR",
        "year",
        "title",
        "journal",
        "source_title",
        "doi",
        "baseline_ToS",
        "in_current_graph",
        "cited_by",
        "internal_indegree",
        "years_since_publication",
        "external_citation_velocity",
        "external_citation_score",
        "internal_citation_score",
        "external_attention_gap_raw",
        "external_attention_gap_score",
        "field_distance",
        "direct_field_neighbor_count",
        "network_proximity_score",
        "text_relevance_raw",
        "text_relevance_score",
        "fruit_score",
        "fruit_signal_type",
    ]
    available_columns = [column for column in output_columns if column in source_df.columns]
    return source_df[available_columns].sort_values(
        ["fruit_score", "external_citation_velocity", "cited_by"],
        ascending=[False, False, False],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank recent fruit candidates from external cited_by and internal network indegree."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument(
        "--user-data-template",
        type=Path,
        default=None,
        help="Optional user_data_template.xlsx path. If omitted, All_Articles_tidy.csv is used.",
    )
    parser.add_argument("--user-data-sheet", default="wos_scopus")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    parser.add_argument("--recent-years", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--field-distance-cutoff", type=int, default=3)
    parser.add_argument(
        "--include-non-main",
        action="store_true",
        help="Include cited-reference-only rows when using BibFusion All_Articles data.",
    )
    parser.add_argument("--w-external-velocity", type=float, default=0.35)
    parser.add_argument("--w-attention-gap", type=float, default=0.35)
    parser.add_argument("--w-network-proximity", type=float, default=0.20)
    parser.add_argument("--w-text-relevance", type=float, default=0.10)
    return parser.parse_args()


def classify_fruit_signal(row: pd.Series) -> str:
    """Describe whether a fruit candidate is still underabsorbed by the tree."""
    if float(row["cited_by"]) <= 0:
        return "recent_no_external_signal"
    if not bool(row["in_current_graph"]):
        return "external_signal_not_matched_in_graph"
    if (
        float(row["internal_indegree"]) <= 1
        and float(row["external_attention_gap_score"]) >= 0.40
    ):
        return "external_fruit_underabsorbed"
    if (
        float(row["internal_indegree"]) >= 5
        and float(row["external_attention_gap_score"]) <= 0.15
    ):
        return "already_absorbing_into_tree"
    return "fruit_candidate"


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    articles_df = read_article_source(
        args.articles,
        args.user_data_template,
        args.user_data_sheet,
    )
    fruit_df = add_fruit_scores(
        graph,
        articles_df,
        args.current_year,
        args.recent_years,
        args.include_non_main,
        args.field_distance_cutoff,
        args.w_external_velocity,
        args.w_attention_gap,
        args.w_network_proximity,
        args.w_text_relevance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "fruit_candidates.csv"
    top_path = args.output_dir / "fruit_candidates_top.csv"
    underabsorbed_path = args.output_dir / "fruit_candidates_underabsorbed.csv"
    fruit_df.to_csv(candidates_path, index=False)
    fruit_df.head(args.top_n).to_csv(top_path, index=False)
    fruit_df[
        fruit_df["fruit_signal_type"] == "external_fruit_underabsorbed"
    ].to_csv(underabsorbed_path, index=False)

    print(f"Fruit candidates: {len(fruit_df):,}")
    print(f"Matched in current graph: {int(fruit_df['in_current_graph'].sum()):,}")
    print(
        "External cited_by range: "
        f"{fruit_df['cited_by'].min():.0f} to {fruit_df['cited_by'].max():.0f}"
    )
    print(
        "Internal indegree range: "
        f"{fruit_df['internal_indegree'].min():.0f} to "
        f"{fruit_df['internal_indegree'].max():.0f}"
    )
    print(f"Saved fruit candidates: {candidates_path.resolve()}")
    print(f"Saved top fruit candidates: {top_path.resolve()}")
    print(f"Saved underabsorbed fruit candidates: {underabsorbed_path.resolve()}")


if __name__ == "__main__":
    main()
