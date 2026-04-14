from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd

from sap import (
    BRANCH,
    ELABORATE_SAP,
    LEAF,
    LEAF_CONNECTIONS,
    RAW_SAP,
    ROOT,
    ROOT_CONNECTIONS,
    SAP,
    TRUNK,
    Sap,
)


DEFAULT_INPUT = Path("data/raw/bibfusion/All_Citation.csv")
DEFAULT_ARTICLES = Path("data/raw/bibfusion/All_Articles.csv")
DEFAULT_OUTPUT = Path("outputs/graphs/citation_network.gexf")
ARTICLE_COLUMNS = [
    "SR",
    "title",
    "author",
    "author_full_names",
    "orcid",
    "abstract",
    "year",
    "journal",
    "source_title",
    "country",
    "doi",
    "__doi_norm",
]


def load_citation_edges(citation_csv: Path) -> pd.DataFrame:
    """Load and clean citation edges from the BibFusion output."""
    df = pd.read_csv(citation_csv, usecols=["SR", "SR_ref"])
    df["SR"] = df["SR"].astype(str).str.strip()
    df["SR_ref"] = df["SR_ref"].astype(str).str.strip()
    df = df[(df["SR"] != "") & (df["SR_ref"] != "")]
    df = df.drop_duplicates()
    return df


def load_article_metadata(articles_csv: Path) -> pd.DataFrame:
    """Load the subset of article metadata used to enrich graph nodes."""
    df = pd.read_csv(articles_csv, usecols=ARTICLE_COLUMNS)
    df["SR"] = df["SR"].astype(str).str.strip()
    df = df[df["SR"] != ""]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = df["year"].apply(lambda value: int(value) if pd.notna(value) else None)
    df = df.drop_duplicates(subset=["SR"], keep="first")
    return df


def build_citation_network(citation_df: pd.DataFrame) -> nx.DiGraph:
    """Create a directed citation graph where SR cites SR_ref."""
    graph = nx.from_pandas_edgelist(
        citation_df,
        source="SR",
        target="SR_ref",
        create_using=nx.DiGraph(),
    )
    return graph


def enrich_network_with_article_data(
    graph: nx.DiGraph, article_df: pd.DataFrame
) -> nx.DiGraph:
    """Attach article metadata to graph nodes using SR as the identifier."""
    enriched_graph = graph.copy()
    article_df = article_df.where(pd.notna(article_df), None)
    node_attributes = article_df.set_index("SR").to_dict(orient="index")

    nx.set_node_attributes(enriched_graph, False, "has_article_metadata")
    nx.set_node_attributes(enriched_graph, "", "title")
    nx.set_node_attributes(enriched_graph, "", "author")
    nx.set_node_attributes(enriched_graph, "", "author_full_names")
    nx.set_node_attributes(enriched_graph, "", "orcid")
    nx.set_node_attributes(enriched_graph, "", "abstract")
    nx.set_node_attributes(enriched_graph, None, "year")
    nx.set_node_attributes(enriched_graph, "", "journal")
    nx.set_node_attributes(enriched_graph, "", "source_title")
    nx.set_node_attributes(enriched_graph, "", "country")
    nx.set_node_attributes(enriched_graph, "", "doi")
    nx.set_node_attributes(enriched_graph, "", "__doi_norm")
    relevant_attributes = {}
    for node, attrs in node_attributes.items():
        if node in enriched_graph:
            attrs["has_article_metadata"] = True
            relevant_attributes[node] = attrs

    nx.set_node_attributes(enriched_graph, relevant_attributes)
    return enriched_graph


def prune_terminal_single_cited_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove nodes with in-degree 1 and out-degree 0."""
    pruned_graph = graph.copy()
    nodes_to_remove = [
        node
        for node in pruned_graph.nodes
        if pruned_graph.in_degree(node) == 1 and pruned_graph.out_degree(node) == 0
    ]
    pruned_graph.remove_nodes_from(nodes_to_remove)
    return pruned_graph


def remove_self_loops(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove edges where a node cites itself."""
    cleaned_graph = graph.copy()
    cleaned_graph.remove_edges_from(nx.selfloop_edges(cleaned_graph))
    return cleaned_graph


def keep_giant_component(graph: nx.DiGraph) -> nx.DiGraph:
    """Keep only the largest weakly connected component of the directed graph."""
    if graph.number_of_nodes() == 0:
        return graph.copy()

    giant_component_nodes = max(nx.weakly_connected_components(graph), key=len)
    return graph.subgraph(giant_component_nodes).copy()


def apply_sap_baseline(graph: nx.DiGraph) -> nx.DiGraph:
    """Apply baseline SAP and keep only user-facing ToS outputs."""
    sap = Sap()
    sap_graph = sap.tree(graph)

    for node, attrs in sap_graph.nodes(data=True):
        attrs["sap_rank"] = attrs.get(SAP, 0)
        if attrs.get(ROOT, 0) > 0:
            attrs["ToS"] = "root"
        elif attrs.get(TRUNK, 0) > 0:
            attrs["ToS"] = "trunk"
        elif attrs.get(LEAF, 0) > 0:
            attrs["ToS"] = "leaf"
        elif attrs.get(BRANCH, 0) > 0:
            attrs["ToS"] = f"branch_{attrs[BRANCH]}"
        else:
            attrs["ToS"] = ""

        for attr_name in (
            ROOT,
            LEAF,
            TRUNK,
            BRANCH,
            SAP,
            ROOT_CONNECTIONS,
            LEAF_CONNECTIONS,
            RAW_SAP,
            ELABORATE_SAP,
        ):
            attrs.pop(attr_name, None)

    return sap_graph


def sanitize_graph_for_gexf(graph: nx.DiGraph) -> nx.DiGraph:
    """Replace None node attributes with empty strings for GEXF export."""
    sanitized_graph = graph.copy()
    for _, attrs in sanitized_graph.nodes(data=True):
        for key, value in list(attrs.items()):
            if value is None:
                attrs[key] = ""
    return sanitized_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a directed citation network from BibFusion All_Citation.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to All_Citation.csv",
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=DEFAULT_ARTICLES,
        help="Path to All_Articles.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the exported GEXF file for Gephi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    citation_df = load_citation_edges(args.input)
    article_df = load_article_metadata(args.articles)
    raw_graph = build_citation_network(citation_df)
    graph_without_self_loops = remove_self_loops(raw_graph)
    cleaned_graph = Sap.clean_graph(graph_without_self_loops)
    graph = enrich_network_with_article_data(cleaned_graph, article_df)
    graph = apply_sap_baseline(graph)
    export_graph = sanitize_graph_for_gexf(graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(export_graph, args.output)

    print(f"Input rows: {len(citation_df):,}")
    print(f"Nodes before pruning: {raw_graph.number_of_nodes():,}")
    print(f"Edges before pruning: {raw_graph.number_of_edges():,}")
    print(
        "Self-loops removed: "
        f"{raw_graph.number_of_edges() - graph_without_self_loops.number_of_edges():,}"
    )
    print(f"Nodes after bibx clean_graph: {graph.number_of_nodes():,}")
    print(f"Edges after bibx clean_graph: {graph.number_of_edges():,}")
    print(
        "Nodes enriched with article metadata: "
        f"{sum(1 for _, attrs in graph.nodes(data=True) if attrs.get('has_article_metadata')):,}"
    )
    print(
        f"ToS-labeled nodes: {sum(1 for _, attrs in graph.nodes(data=True) if attrs.get('ToS')):,}"
    )
    print(f"Is directed: {graph.is_directed()}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
