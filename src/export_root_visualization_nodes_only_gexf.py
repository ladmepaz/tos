from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


DEFAULT_NODES_INPUT = Path("outputs/root_visualization/root_visualization_metrics.csv")
DEFAULT_OUTPUT = Path("outputs/root_visualization/gephi/root_visualization_nodes_only.gexf")


def hex_to_rgba(hex_color: str) -> dict[str, float]:
    """Convert a hex color to a Gephi viz color dict."""
    clean = hex_color.lstrip("#")
    if len(clean) != 6:
        return {"r": 200, "g": 200, "b": 200, "a": 1.0}
    return {
        "r": int(clean[0:2], 16),
        "g": int(clean[2:4], 16),
        "b": int(clean[4:6], 16),
        "a": 1.0,
    }


def build_nodes_only_graph(nodes_df: pd.DataFrame) -> nx.Graph:
    """Build a root-only graph with styled nodes and no edges."""
    graph = nx.Graph()

    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            row.node_id,
            label=(row.title if isinstance(row.title, str) and row.title.strip() else row.node_id),
            subtopic_id=row.subtopic_id,
            subtopic_label=row.subtopic_label,
            cluster_order=int(row.cluster_order),
            cluster_strength=float(row.cluster_strength),
            cluster_cohesion=float(row.cluster_cohesion),
            cluster_separation=float(row.cluster_separation),
            cluster_support=float(row.cluster_support),
            is_representative=bool(row.is_representative),
            root_anchor_score=float(row.root_anchor_score),
            sap_rank=float(row.sap_rank),
            year="" if pd.isna(row.year) else int(row.year),
            journal="" if pd.isna(row.journal) else str(row.journal),
            source_title="" if pd.isna(row.source_title) else str(row.source_title),
            has_article_metadata=bool(row.has_article_metadata),
            citations_received=float(row.citations_received),
            citation_velocity=float(row.citation_velocity),
            citation_velocity_score=float(row.citation_velocity_score),
            global_size_score=float(row.global_size_score),
            local_size_score=float(row.local_size_score),
            final_root_size_score=float(row.final_root_size_score),
            color=str(row.color),
            viz={
                "size": float(row.size),
                "color": hex_to_rgba(str(row.color)),
            },
        )

    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a nodes-only root GEXF with Gephi viz color and size."
    )
    parser.add_argument("--nodes-input", type=Path, default=DEFAULT_NODES_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_csv(args.nodes_input)
    graph = build_nodes_only_graph(nodes_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, args.output)

    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")
    print(f"Saved GEXF: {args.output.resolve()}")


if __name__ == "__main__":
    main()
