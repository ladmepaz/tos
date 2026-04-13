from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


DEFAULT_INPUT = Path("data/raw/bibfusion/All_Citation.csv")
DEFAULT_OUTPUT = Path("outputs/graphs/citation_network.gexf")


def load_citation_edges(citation_csv: Path) -> pd.DataFrame:
    """Load and clean citation edges from the BibFusion output."""
    df = pd.read_csv(citation_csv, usecols=["SR", "SR_ref"])
    df["SR"] = df["SR"].astype(str).str.strip()
    df["SR_ref"] = df["SR_ref"].astype(str).str.strip()
    df = df[(df["SR"] != "") & (df["SR_ref"] != "")]
    df = df.drop_duplicates()
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the exported GEXF file for Gephi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    citation_df = load_citation_edges(args.input)
    raw_graph = build_citation_network(citation_df)
    graph_without_self_loops = remove_self_loops(raw_graph)
    pruned_graph = prune_terminal_single_cited_nodes(graph_without_self_loops)
    graph = keep_giant_component(pruned_graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, args.output)

    print(f"Input rows: {len(citation_df):,}")
    print(f"Nodes before pruning: {raw_graph.number_of_nodes():,}")
    print(f"Edges before pruning: {raw_graph.number_of_edges():,}")
    print(
        "Self-loops removed: "
        f"{raw_graph.number_of_edges() - graph_without_self_loops.number_of_edges():,}"
    )
    print(f"Nodes after pruning: {pruned_graph.number_of_nodes():,}")
    print(f"Edges after pruning: {pruned_graph.number_of_edges():,}")
    print(f"Nodes in giant component: {graph.number_of_nodes():,}")
    print(f"Edges in giant component: {graph.number_of_edges():,}")
    print(f"Is directed: {graph.is_directed()}")
    print(f"Saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
