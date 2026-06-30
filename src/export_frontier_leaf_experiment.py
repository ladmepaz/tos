from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_citation_network import (
    build_citation_network,
    enrich_network_with_article_data,
    load_article_metadata,
    load_citation_edges,
    remove_self_loops,
)
from export_leaf_visualization_svg import load_top_fruit_ids
from sap import BRANCH, LEAF, ROOT, SAP, TRUNK, Sap


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_FRUITS_TOP = Path("outputs/fruit_visualization/fruits_top3.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/frontier_leaf_experiment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine SAP leaves with recent frontier leaves up to a target count."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--fruits-top", type=Path, default=DEFAULT_FRUITS_TOP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-leaves", type=int, default=100)
    parser.add_argument("--recent-window-years", type=int, default=5)
    return parser.parse_args()


def build_enriched_graph(citation_csv: Path, articles_csv: Path):
    citation_df = load_citation_edges(citation_csv)
    article_df = load_article_metadata(articles_csv)
    graph = build_citation_network(citation_df)
    graph = remove_self_loops(graph)
    graph = Sap.clean_graph(graph)
    return enrich_network_with_article_data(graph, article_df)


def load_article_lookup(path: Path) -> dict[str, dict]:
    df = pd.read_csv(path)
    df["SR"] = df["SR"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cited_by"] = pd.to_numeric(df.get("cited_by"), errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["SR"], keep="first")
    return df.set_index("SR").to_dict(orient="index")


def run_sap(graph, max_leaves: int):
    return Sap(max_leaves=max_leaves).tree(graph)


def extract_sap_leaves(graph, target_leaves: int, article_lookup: dict[str, dict]) -> pd.DataFrame:
    rows = []
    leaves = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if attrs.get(LEAF, 0) > 0
    ]
    leaves = sorted(
        leaves,
        key=lambda item: (item[1].get(LEAF, 0), item[1].get(SAP, 0), str(item[0])),
        reverse=True,
    )
    for rank, (node, attrs) in enumerate(leaves, start=1):
        extra = article_lookup.get(str(node), {})
        rows.append(
            node_record(
                graph,
                node,
                attrs,
                extra,
                leaf_rank=rank,
                leaf_type="sap_leaf",
                target_leaves=target_leaves,
                frontier_rank=None,
            )
        )
    return pd.DataFrame(rows)


def node_record(
    graph,
    node: str,
    attrs: dict,
    extra: dict,
    leaf_rank: int,
    leaf_type: str,
    target_leaves: int,
    frontier_rank: int | None,
) -> dict:
    return {
        "leaf_rank": leaf_rank,
        "frontier_rank": frontier_rank,
        "leaf_type": leaf_type,
        "target_leaves": target_leaves,
        "node_id": node,
        "title": attrs.get("title") or extra.get("title") or "",
        "abstract": attrs.get("abstract") or extra.get("abstract") or "",
        "year": attrs.get("year") if attrs.get("year") is not None else extra.get("year"),
        "journal": attrs.get("journal") or extra.get("journal") or "",
        "source_title": attrs.get("source_title") or extra.get("source_title") or "",
        "doi": attrs.get("doi") or extra.get("doi") or "",
        "cited_by": extra.get("cited_by", 0),
        "has_article_metadata": attrs.get("has_article_metadata", False),
        "leaf_connection_score": attrs.get(LEAF, 0),
        "sap_rank": attrs.get(SAP, 0),
        "internal_indegree": graph.in_degree(node),
        "internal_outdegree": graph.out_degree(node),
    }


def extract_frontier_candidates(
    graph,
    article_lookup: dict[str, dict],
    sap_leaf_ids: set[str],
    fruit_ids: set[str],
    recent_window_years: int,
) -> pd.DataFrame:
    years = [
        attrs.get("year")
        for _, attrs in graph.nodes(data=True)
        if attrs.get("year") is not None
    ]
    newest_year = int(max(years))
    earliest_year = newest_year - recent_window_years + 1

    rows = []
    for node, attrs in graph.nodes(data=True):
        node_id = str(node)
        year = attrs.get("year")
        if node_id in sap_leaf_ids or node_id in fruit_ids:
            continue
        if attrs.get(ROOT, 0) > 0 or attrs.get(TRUNK, 0) > 0 or attrs.get(BRANCH, 0) > 0:
            continue
        if year is None or pd.isna(year) or int(year) < earliest_year:
            continue
        if graph.in_degree(node) != 0 or graph.out_degree(node) < 1:
            continue

        extra = article_lookup.get(node_id, {})
        rows.append(
            node_record(
                graph,
                node_id,
                attrs,
                extra,
                leaf_rank=0,
                leaf_type="frontier_leaf_candidate",
                target_leaves=0,
                frontier_rank=None,
            )
        )

    candidate_df = pd.DataFrame(rows)
    if candidate_df.empty:
        return candidate_df

    candidate_df["year"] = pd.to_numeric(candidate_df["year"], errors="coerce")
    candidate_df["cited_by"] = pd.to_numeric(candidate_df["cited_by"], errors="coerce").fillna(0)
    candidate_df["has_title_abstract"] = (
        candidate_df["title"].fillna("").astype(str).str.len()
        + candidate_df["abstract"].fillna("").astype(str).str.len()
    ) > 0
    return candidate_df.sort_values(
        [
            "year",
            "internal_outdegree",
            "cited_by",
            "has_title_abstract",
            "node_id",
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def build_combined_leaves(
    sap_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    target_leaves: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill_count = max(0, target_leaves - len(sap_df))
    selected_frontier_df = candidates_df.head(fill_count).copy()
    if not selected_frontier_df.empty:
        selected_frontier_df["frontier_rank"] = range(1, len(selected_frontier_df) + 1)
        selected_frontier_df["leaf_type"] = "frontier_leaf"
        selected_frontier_df["target_leaves"] = target_leaves
        selected_frontier_df["leaf_rank"] = range(len(sap_df) + 1, len(sap_df) + len(selected_frontier_df) + 1)

    combined_df = pd.concat([sap_df, selected_frontier_df], ignore_index=True)
    return combined_df, selected_frontier_df


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Frontier Leaf Experiment",
        "",
        "This list keeps SAP leaves first and fills the remaining slots with recent frontier leaves.",
        "",
    ]
    for row in df.itertuples(index=False):
        year = "" if pd.isna(row.year) else str(int(float(row.year)))
        title = str(row.title).strip() or "No title"
        lines.append(
            f"{int(row.leaf_rank)}. **{row.node_id}** ({year}) - {title} "
            f"[{row.leaf_type}; outdegree={int(row.internal_outdegree)}; cited_by={float(row.cited_by):.0f}]"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(
    sap_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    selected_frontier_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    recent_window_years: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_leaves": int(combined_df["target_leaves"].max()) if not combined_df.empty else 0,
                "sap_leaf_count": len(sap_df),
                "frontier_candidate_count": len(candidates_df),
                "selected_frontier_leaf_count": len(selected_frontier_df),
                "combined_leaf_count": len(combined_df),
                "recent_window_years": recent_window_years,
                "selected_frontier_min_year": int(selected_frontier_df["year"].min())
                if not selected_frontier_df.empty
                else None,
                "selected_frontier_max_year": int(selected_frontier_df["year"].max())
                if not selected_frontier_df.empty
                else None,
                "selected_frontier_min_outdegree": int(selected_frontier_df["internal_outdegree"].min())
                if not selected_frontier_df.empty
                else None,
                "selected_frontier_max_outdegree": int(selected_frontier_df["internal_outdegree"].max())
                if not selected_frontier_df.empty
                else None,
            }
        ]
    )


def main() -> None:
    args = parse_args()
    graph = build_enriched_graph(args.citations, args.articles)
    article_lookup = load_article_lookup(args.articles)
    sap_graph = run_sap(graph, args.target_leaves)
    fruit_ids = load_top_fruit_ids(args.fruits_top)

    sap_df = extract_sap_leaves(sap_graph, args.target_leaves, article_lookup)
    candidates_df = extract_frontier_candidates(
        sap_graph,
        article_lookup,
        set(sap_df["node_id"]),
        fruit_ids,
        args.recent_window_years,
    )
    combined_df, selected_frontier_df = build_combined_leaves(
        sap_df,
        candidates_df,
        args.target_leaves,
    )
    summary_df = summarize(
        sap_df,
        candidates_df,
        selected_frontier_df,
        combined_df,
        args.recent_window_years,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sap_path = args.output_dir / "sap_leaves.csv"
    candidate_path = args.output_dir / "frontier_leaf_candidates.csv"
    selected_path = args.output_dir / "selected_frontier_leaves.csv"
    combined_path = args.output_dir / "combined_leaves_top_100.csv"
    markdown_path = args.output_dir / "combined_leaves_top_100.md"
    summary_path = args.output_dir / "frontier_leaf_summary.csv"

    sap_df.to_csv(sap_path, index=False)
    candidates_df.to_csv(candidate_path, index=False)
    selected_frontier_df.to_csv(selected_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_markdown(combined_df, markdown_path)

    print(f"SAP leaves: {len(sap_df):,}")
    print(f"Frontier candidates: {len(candidates_df):,}")
    print(f"Selected frontier leaves: {len(selected_frontier_df):,}")
    print(f"Combined leaves: {len(combined_df):,}")
    print(f"Saved combined markdown: {markdown_path.resolve()}")
    print(f"Saved selected frontier leaves: {selected_path.resolve()}")
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
