from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


DEFAULT_GRAPH = Path("outputs/graphs/citation_network.gexf")
DEFAULT_OUTPUT = Path("outputs/graphs/tos_classification_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 03: validate and summarize SAP/ToS labels. SAP is applied by "
            "src/build_citation_network.py in the current implementation."
        )
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = nx.read_gexf(args.graph)
    rows = []
    for node, attrs in graph.nodes(data=True):
        tos_label = str(attrs.get("ToS", "")).strip() or "unclassified"
        rows.append(
            {
                "node_id": node,
                "ToS": tos_label,
                "sap_rank": attrs.get("sap_rank", ""),
            }
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("ToS", dropna=False)
        .size()
        .reset_index(name="node_count")
        .sort_values(["ToS"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

