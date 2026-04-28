from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from root_tfidf_similarity import build_sap_graph, extract_root_records

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_structural")


def compute_root_influence_sets(
    graph: nx.DiGraph, records: list[dict]
) -> dict[str, set[str]]:
    """Collect downstream influence sets for each root.

    With the current edge convention (citing -> cited), a root's downstream
    influenced papers are the graph ancestors that can reach the root.
    """
    influence_sets = {}
    for record in records:
        influence_sets[record["root_id"]] = nx.ancestors(graph, record["node_id"])
    return influence_sets


def overlap_cosine(nodes_a: set[str], nodes_b: set[str]) -> float:
    """Compute Salton cosine from set overlap."""
    if not nodes_a or not nodes_b:
        return 0.0
    return len(nodes_a & nodes_b) / math.sqrt(len(nodes_a) * len(nodes_b))


def overlap_jaccard(nodes_a: set[str], nodes_b: set[str]) -> float:
    """Compute Jaccard similarity from set overlap."""
    if not nodes_a and not nodes_b:
        return 0.0
    union_size = len(nodes_a | nodes_b)
    if union_size == 0:
        return 0.0
    return len(nodes_a & nodes_b) / union_size


def build_structural_outputs(
    records: list[dict],
    influence_sets: dict[str, set[str]],
):
    """Build root table, structural matrix, and pairwise overlap outputs."""
    root_ids = [record["root_id"] for record in records]
    node_by_root_id = {record["root_id"]: record["node_id"] for record in records}
    title_by_root_id = {record["root_id"]: record["title"] for record in records}

    matrix_rows = []
    edge_rows = []
    for source_id in root_ids:
        matrix_row = {"root_id": source_id}
        for target_id in root_ids:
            cosine = overlap_cosine(
                influence_sets[source_id], influence_sets[target_id]
            )
            matrix_row[target_id] = round(cosine, 6)
            if source_id < target_id:
                shared_nodes = influence_sets[source_id] & influence_sets[target_id]
                edge_rows.append(
                    {
                        "source_root_id": source_id,
                        "target_root_id": target_id,
                        "source_node_id": node_by_root_id[source_id],
                        "target_node_id": node_by_root_id[target_id],
                        "source_title": title_by_root_id[source_id],
                        "target_title": title_by_root_id[target_id],
                        "source_influence_count": len(influence_sets[source_id]),
                        "target_influence_count": len(influence_sets[target_id]),
                        "shared_influence_count": len(shared_nodes),
                        "structural_cosine": round(cosine, 6),
                        "structural_jaccard": round(
                            overlap_jaccard(
                                influence_sets[source_id], influence_sets[target_id]
                            ),
                            6,
                        ),
                    }
                )
        matrix_rows.append(matrix_row)

    root_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "tokens"}
        row["influence_count"] = len(influence_sets[record["root_id"]])
        root_rows.append(row)

    root_df = pd.DataFrame(root_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    edges_df = pd.DataFrame(edge_rows).sort_values(
        ["structural_cosine", "shared_influence_count"],
        ascending=[False, False],
    )
    return root_df, matrix_df, edges_df


def build_similarity_graph(
    records: list[dict],
    influence_sets: dict[str, set[str]],
    min_similarity: float,
) -> nx.Graph:
    """Create a weighted root-only graph from structural cosine similarity."""
    graph = nx.Graph()
    for record in records:
        graph.add_node(
            record["root_id"],
            node_id=record["node_id"],
            sap_rank=record["sap_rank"],
            title=record["title"],
        )

    root_ids = [record["root_id"] for record in records]
    for index, source_id in enumerate(root_ids):
        for target_id in root_ids[index + 1 :]:
            cosine = overlap_cosine(
                influence_sets[source_id], influence_sets[target_id]
            )
            if cosine >= min_similarity:
                graph.add_edge(source_id, target_id, weight=cosine)
    return graph


def summarize_shared_influence(
    community: set[str],
    influence_sets: dict[str, set[str]],
    top_n: int = 8,
) -> list[str]:
    """Summarize a structural community by its most recurrent influenced nodes."""
    counter: dict[str, int] = {}
    for root_id in community:
        for node_id in influence_sets[root_id]:
            counter[node_id] = counter.get(node_id, 0) + 1
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [node_id for node_id, _ in ordered[:top_n]]


def detect_subtopics(
    records: list[dict],
    influence_sets: dict[str, set[str]],
    min_similarity: float,
):
    """Group root papers into structural-overlap subtopics."""
    similarity_graph = build_similarity_graph(records, influence_sets, min_similarity)
    if similarity_graph.number_of_edges() == 0:
        communities = [{record["root_id"]} for record in records]
    else:
        communities = list(
            nx.community.greedy_modularity_communities(
                similarity_graph,
                weight="weight",
            )
        )
        assigned_nodes = set().union(*communities) if communities else set()
        missing_nodes = set(similarity_graph.nodes) - assigned_nodes
        communities.extend({node} for node in missing_nodes)

    root_lookup = {record["root_id"]: record for record in records}
    sorted_communities = sorted(
        communities,
        key=lambda community: (
            -len(community),
            -max(root_lookup[root_id]["sap_rank"] for root_id in community),
        ),
    )

    subtopic_rows = []
    summary_rows = []
    for index, community in enumerate(sorted_communities, start=1):
        subtopic_id = f"subtopic_{index}"
        representative_root_id = max(
            community,
            key=lambda root_id: root_lookup[root_id]["sap_rank"],
        )
        representative_node_id = root_lookup[representative_root_id]["node_id"]
        shared_influence = summarize_shared_influence(community, influence_sets)
        subtopic_label = (
            "; ".join(shared_influence[:3]) if shared_influence else "isolated_root"
        )

        member_root_ids = sorted(
            community,
            key=lambda root_id: root_lookup[root_id]["sap_rank"],
            reverse=True,
        )
        member_node_ids = [root_lookup[root_id]["node_id"] for root_id in member_root_ids]

        summary_rows.append(
            {
                "subtopic_id": subtopic_id,
                "subtopic_label": subtopic_label,
                "member_count": len(community),
                "representative_root_id": representative_root_id,
                "representative_node_id": representative_node_id,
                "top_shared_influence_nodes": "; ".join(shared_influence),
                "member_root_ids": "; ".join(member_root_ids),
                "member_node_ids": "; ".join(member_node_ids),
            }
        )

        for root_id in member_root_ids:
            record = root_lookup[root_id]
            subtopic_rows.append(
                {
                    "subtopic_id": subtopic_id,
                    "subtopic_label": subtopic_label,
                    "is_representative": root_id == representative_root_id,
                    "root_id": root_id,
                    "node_id": record["node_id"],
                    "sap_rank": record["sap_rank"],
                    "title": record["title"],
                    "year": record["year"],
                    "journal": record["journal"],
                    "source_title": record["source_title"],
                    "has_article_metadata": record["has_article_metadata"],
                    "influence_count": len(influence_sets[root_id]),
                }
            )

    subtopics_df = pd.DataFrame(subtopic_rows).sort_values(
        ["subtopic_id", "sap_rank"],
        ascending=[True, False],
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("subtopic_id")
    return similarity_graph, subtopics_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute structural similarity and subtopics among ToS root papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.20,
        help="Minimum structural cosine to connect two roots in the subtopic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    influence_sets = compute_root_influence_sets(graph, records)
    root_df, matrix_df, edges_df = build_structural_outputs(records, influence_sets)
    similarity_graph, subtopics_df, summary_df = detect_subtopics(
        records,
        influence_sets,
        args.min_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_df.to_csv(args.output_dir / "root_papers.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_structural_matrix.csv", index=False)
    edges_df.to_csv(args.output_dir / "root_structural_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_structural_subtopics.csv", index=False)
    summary_df.to_csv(
        args.output_dir / "root_structural_subtopic_summary.csv",
        index=False,
    )

    print(f"Root papers: {len(root_df):,}")
    print(f"Structural pairs: {len(edges_df):,}")
    print(f"Subtopics: {len(summary_df):,}")
    print(
        "Subtopic similarity edges: "
        f"{similarity_graph.number_of_edges():,} (threshold={args.min_similarity:.2f})"
    )
    print(f"Saved root papers: {(args.output_dir / 'root_papers.csv').resolve()}")
    print(
        "Saved structural matrix: "
        f"{(args.output_dir / 'root_structural_matrix.csv').resolve()}"
    )
    print(
        "Saved structural pairs: "
        f"{(args.output_dir / 'root_structural_pairs.csv').resolve()}"
    )
    print(
        "Saved structural subtopics: "
        f"{(args.output_dir / 'root_structural_subtopics.csv').resolve()}"
    )
    print(
        "Saved structural subtopic summary: "
        f"{(args.output_dir / 'root_structural_subtopic_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
