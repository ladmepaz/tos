from __future__ import annotations

import argparse
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from networkx.algorithms.community.louvain import louvain_communities

from root_tfidf_similarity import build_sap_graph, cosine_similarity, tokenize
from sap import ROOT, TRUNK, YEAR


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/branch_leiden_comparison")


def require_leiden() -> tuple[object, object]:
    """Import Leiden dependencies only when this experimental script is used."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise SystemExit(
            "Leiden support requires optional packages. Install them with:\n"
            "python -m pip install igraph leidenalg"
        ) from exc
    return ig, leidenalg


def nx_to_igraph(graph: nx.Graph):
    """Convert a NetworkX graph to igraph while preserving node labels."""
    ig, _ = require_leiden()
    nodes = sorted(graph.nodes, key=str)
    node_index = {node: index for index, node in enumerate(nodes)}
    edges = [(node_index[source], node_index[target]) for source, target in graph.edges]
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.vs["name"] = [str(node) for node in nodes]
    return ig_graph, nodes


def detect_louvain_communities(graph: nx.Graph, seed: int) -> list[set[str]]:
    """Detect communities with the current SAP baseline algorithm."""
    return [set(community) for community in louvain_communities(graph, seed=seed)]


def detect_leiden_communities(graph: nx.Graph, seed: int) -> list[set[str]]:
    """Detect communities with Leiden using modularity for Louvain comparability."""
    _, leidenalg = require_leiden()
    ig_graph, nodes = nx_to_igraph(graph)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        seed=seed,
    )
    return [{nodes[index] for index in community} for community in partition]


def sort_communities(communities: Iterable[set[str]]) -> list[set[str]]:
    """Match the current SAP branch ordering rule for fair comparison."""
    return sorted(communities, key=lambda community: (len(community), sorted(community)[0]))


def branch_node_priority(graph: nx.DiGraph, node: str) -> tuple[float, float, str]:
    """Rank branch candidates by recency first and SAP relevance second."""
    attrs = graph.nodes[node]
    return (
        float(attrs.get(YEAR) or 0),
        float(attrs.get("sap_rank") or attrs.get("_sap") or 0),
        str(node),
    )


def eligible_branch_nodes(graph: nx.DiGraph, community: set[str]) -> list[str]:
    """Return nodes that can be selected as branch papers."""
    return [
        node
        for node in community
        if graph.nodes[node].get(ROOT, 0) == 0
        and graph.nodes[node].get(TRUNK, 0) == 0
        and graph.nodes[node].get(YEAR) is not None
    ]


def select_current_branch_nodes(
    graph: nx.DiGraph,
    community: set[str],
    max_branch_size: int,
) -> list[str]:
    """Select branch papers with the current SAP-compatible recency rule."""
    return sorted(
        eligible_branch_nodes(graph, community),
        key=lambda node: (graph.nodes[node].get(YEAR) or 0, str(node)),
        reverse=True,
    )[:max_branch_size]


def greedy_connected_subset(
    graph: nx.Graph,
    component_nodes: set[str],
    max_branch_size: int,
) -> list[str]:
    """Build a connected branch subset by expanding from the strongest recent paper."""
    if not component_nodes:
        return []

    seed = max(component_nodes, key=lambda node: branch_node_priority(graph, node))
    selected = [seed]
    selected_set = {seed}

    while len(selected) < max_branch_size:
        frontier = {
            neighbor
            for node in selected_set
            for neighbor in graph.neighbors(node)
            if neighbor in component_nodes and neighbor not in selected_set
        }
        if not frontier:
            break
        next_node = max(frontier, key=lambda node: branch_node_priority(graph, node))
        selected.append(next_node)
        selected_set.add(next_node)

    return selected


def select_cohesive_branch_nodes(
    graph: nx.DiGraph,
    community: set[str],
    max_branch_size: int,
) -> list[str]:
    """Select branch papers while preserving a connected induced branch subset."""
    undirected = graph.to_undirected()
    candidate_nodes = eligible_branch_nodes(graph, community)
    candidate_subgraph = undirected.subgraph(candidate_nodes).copy()
    components = [set(component) for component in nx.connected_components(candidate_subgraph)]
    if not components:
        return []

    component_options = []
    for component in components:
        selected = greedy_connected_subset(undirected, component, max_branch_size)
        component_options.append(
            (
                selected,
                len(selected),
                sum(branch_node_priority(graph, node)[0] for node in selected),
                sum(branch_node_priority(graph, node)[1] for node in selected),
                max(branch_node_priority(graph, node) for node in selected),
            )
        )

    best_selected, *_ = max(
        component_options,
        key=lambda option: (option[1], option[2], option[3], option[4]),
    )
    return best_selected


def select_branch_members(
    graph: nx.DiGraph,
    communities: list[set[str]],
    max_branches: int,
    max_branch_size: int,
    selection_strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select branch papers from communities using a named selection strategy."""
    community_rows = []
    member_rows = []
    selected_communities = sort_communities(communities)[:max_branches]

    for branch_index, community in enumerate(selected_communities, start=1):
        branch_label = f"branch_{branch_index}"
        eligible_nodes = eligible_branch_nodes(graph, community)
        if selection_strategy == "current":
            candidate_nodes = select_current_branch_nodes(graph, community, max_branch_size)
        elif selection_strategy == "cohesive":
            candidate_nodes = select_cohesive_branch_nodes(graph, community, max_branch_size)
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        community_rows.append(
            {
                "branch_label": branch_label,
                "selection_strategy": selection_strategy,
                "community_size": len(community),
                "eligible_candidate_count": len(eligible_nodes),
                "candidate_count": len(candidate_nodes),
                "community_node_ids": "; ".join(sorted(map(str, community))),
            }
        )

        for rank, node in enumerate(candidate_nodes, start=1):
            attrs = graph.nodes[node]
            member_rows.append(
                {
                    "branch_label": branch_label,
                    "selection_strategy": selection_strategy,
                    "branch_rank": rank,
                    "node_id": node,
                    "baseline_ToS": attrs.get("ToS", ""),
                    "sap_rank": attrs.get("sap_rank", 0),
                    "year": attrs.get("year") or "",
                    "title": attrs.get("title") or "",
                    "journal": attrs.get("journal") or "",
                    "source_title": attrs.get("source_title") or "",
                    "doi": attrs.get("doi") or "",
                    "has_article_metadata": attrs.get("has_article_metadata", False),
                }
            )

    return pd.DataFrame(member_rows), pd.DataFrame(community_rows)


def compute_tfidf_vectors(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute TF-IDF vectors for selected branch papers."""
    document_tokens = {record["node_id"]: record["tokens"] for record in records}
    document_count = len(document_tokens)
    document_frequency = Counter()
    for tokens in document_tokens.values():
        document_frequency.update(set(tokens))

    vectors = {}
    for node_id, tokens in document_tokens.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        vector = {}
        if total_terms:
            for term, count in term_counts.items():
                term_frequency = count / total_terms
                inverse_document_frequency = math.log(
                    (1 + document_count) / (1 + document_frequency[term])
                ) + 1
                vector[term] = term_frequency * inverse_document_frequency
        vectors[node_id] = vector
    return vectors


def mean_pairwise_text_similarity(graph: nx.DiGraph, node_ids: list[str]) -> float:
    """Summarize semantic cohesion of a branch with title/abstract similarity."""
    if len(node_ids) < 2:
        return 0.0
    records = []
    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        tokens = tokenize(f"{attrs.get('title') or ''} {attrs.get('abstract') or ''}")
        records.append({"node_id": node_id, "tokens": tokens})
    vectors = compute_tfidf_vectors(records)
    similarities = [
        cosine_similarity(vectors[source], vectors[target])
        for source, target in combinations(node_ids, 2)
    ]
    return sum(similarities) / len(similarities) if similarities else 0.0


def summarize_branch_quality(
    graph: nx.DiGraph,
    members_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    algorithm: str,
) -> pd.DataFrame:
    """Compute structural and text-quality diagnostics for selected branches."""
    undirected = graph.to_undirected()
    rows = []
    for row in communities_df.itertuples(index=False):
        community_node_ids = [
            node_id for node_id in str(row.community_node_ids).split("; ") if node_id
        ]
        community_subgraph = undirected.subgraph(community_node_ids).copy()
        community_component_sizes = (
            [len(component) for component in nx.connected_components(community_subgraph)]
            if community_node_ids
            else []
        )
        community_largest_component = (
            max(community_component_sizes) if community_component_sizes else 0
        )

        branch_members = members_df[members_df["branch_label"] == row.branch_label]
        node_ids = branch_members["node_id"].astype(str).tolist()
        subgraph = undirected.subgraph(node_ids).copy()
        component_sizes = (
            [len(component) for component in nx.connected_components(subgraph)]
            if node_ids
            else []
        )
        largest_component = max(component_sizes) if component_sizes else 0
        rows.append(
            {
                "algorithm": algorithm,
                "branch_label": row.branch_label,
                "selection_strategy": row.selection_strategy,
                "community_size": row.community_size,
                "eligible_candidate_count": row.eligible_candidate_count,
                "community_internal_edges": community_subgraph.number_of_edges(),
                "community_density": round(nx.density(community_subgraph), 6)
                if len(community_node_ids) > 1
                else 0.0,
                "community_connected_components": len(community_component_sizes),
                "community_largest_component_fraction": round(
                    community_largest_component / len(community_node_ids),
                    6,
                )
                if community_node_ids
                else 0.0,
                "selected_node_count": len(node_ids),
                "internal_edges_selected": subgraph.number_of_edges(),
                "density_selected": round(nx.density(subgraph), 6) if len(node_ids) > 1 else 0.0,
                "connected_components_selected": len(component_sizes),
                "largest_component_fraction": round(largest_component / len(node_ids), 6)
                if node_ids
                else 0.0,
                "mean_pairwise_text_similarity": round(
                    mean_pairwise_text_similarity(graph, node_ids),
                    6,
                ),
                "member_node_ids": "; ".join(node_ids),
            }
        )
    return pd.DataFrame(rows)


def build_overlap_table(
    louvain_members: pd.DataFrame,
    leiden_members: pd.DataFrame,
) -> pd.DataFrame:
    """Show how selected branch papers move between Louvain and Leiden."""
    louvain_lookup = dict(zip(louvain_members["node_id"], louvain_members["branch_label"]))
    leiden_lookup = dict(zip(leiden_members["node_id"], leiden_members["branch_label"]))
    node_ids = sorted(set(louvain_lookup) | set(leiden_lookup))
    return pd.DataFrame(
        [
            {
                "node_id": node_id,
                "louvain_branch": louvain_lookup.get(node_id, ""),
                "leiden_branch": leiden_lookup.get(node_id, ""),
                "same_branch_label": louvain_lookup.get(node_id, "")
                == leiden_lookup.get(node_id, ""),
            }
            for node_id in node_ids
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Louvain and Leiden for experimental ToS branch detection."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-branches", type=int, default=3)
    parser.add_argument("--max-branch-size", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    undirected = graph.to_undirected()

    louvain_communities_result = detect_louvain_communities(undirected, args.seed)
    leiden_communities_result = detect_leiden_communities(undirected, args.seed)

    louvain_members, louvain_community_summary = select_branch_members(
        graph,
        louvain_communities_result,
        args.max_branches,
        args.max_branch_size,
        "current",
    )
    leiden_members, leiden_community_summary = select_branch_members(
        graph,
        leiden_communities_result,
        args.max_branches,
        args.max_branch_size,
        "current",
    )
    louvain_cohesive_members, louvain_cohesive_community_summary = select_branch_members(
        graph,
        louvain_communities_result,
        args.max_branches,
        args.max_branch_size,
        "cohesive",
    )
    leiden_cohesive_members, leiden_cohesive_community_summary = select_branch_members(
        graph,
        leiden_communities_result,
        args.max_branches,
        args.max_branch_size,
        "cohesive",
    )

    louvain_quality = summarize_branch_quality(
        graph,
        louvain_members,
        louvain_community_summary,
        "louvain_current",
    )
    leiden_quality = summarize_branch_quality(
        graph,
        leiden_members,
        leiden_community_summary,
        "leiden_current",
    )
    louvain_cohesive_quality = summarize_branch_quality(
        graph,
        louvain_cohesive_members,
        louvain_cohesive_community_summary,
        "louvain_cohesive",
    )
    leiden_cohesive_quality = summarize_branch_quality(
        graph,
        leiden_cohesive_members,
        leiden_cohesive_community_summary,
        "leiden_cohesive",
    )
    quality_df = pd.concat(
        [
            louvain_quality,
            leiden_quality,
            louvain_cohesive_quality,
            leiden_cohesive_quality,
        ],
        ignore_index=True,
    )
    overlap_df = build_overlap_table(louvain_members, leiden_members)
    cohesive_overlap_df = build_overlap_table(
        louvain_cohesive_members,
        leiden_cohesive_members,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    louvain_members.to_csv(args.output_dir / "louvain_branch_members.csv", index=False)
    leiden_members.to_csv(args.output_dir / "leiden_branch_members.csv", index=False)
    louvain_members.to_csv(
        args.output_dir / "louvain_current_branch_members.csv",
        index=False,
    )
    leiden_members.to_csv(
        args.output_dir / "leiden_current_branch_members.csv",
        index=False,
    )
    louvain_cohesive_members.to_csv(
        args.output_dir / "louvain_cohesive_branch_members.csv",
        index=False,
    )
    leiden_cohesive_members.to_csv(
        args.output_dir / "leiden_cohesive_branch_members.csv",
        index=False,
    )
    louvain_community_summary.to_csv(
        args.output_dir / "louvain_branch_communities.csv",
        index=False,
    )
    leiden_community_summary.to_csv(
        args.output_dir / "leiden_branch_communities.csv",
        index=False,
    )
    louvain_cohesive_community_summary.to_csv(
        args.output_dir / "louvain_cohesive_branch_communities.csv",
        index=False,
    )
    leiden_cohesive_community_summary.to_csv(
        args.output_dir / "leiden_cohesive_branch_communities.csv",
        index=False,
    )
    quality_df.to_csv(args.output_dir / "louvain_vs_leiden_branch_quality.csv", index=False)
    quality_df.to_csv(args.output_dir / "branch_selection_quality_comparison.csv", index=False)
    overlap_df.to_csv(args.output_dir / "louvain_leiden_node_overlap.csv", index=False)
    cohesive_overlap_df.to_csv(
        args.output_dir / "louvain_leiden_cohesive_node_overlap.csv",
        index=False,
    )

    print(f"Louvain selected branch papers: {len(louvain_members):,}")
    print(f"Leiden selected branch papers: {len(leiden_members):,}")
    print(f"Louvain cohesive branch papers: {len(louvain_cohesive_members):,}")
    print(f"Leiden cohesive branch papers: {len(leiden_cohesive_members):,}")
    print(f"Overlap rows: {len(overlap_df):,}")
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
