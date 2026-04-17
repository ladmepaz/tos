from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from root_tfidf_similarity import build_sap_graph, extract_root_records

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_cocitation")


def compute_root_citers(
    graph: nx.DiGraph, records: list[dict]
) -> dict[str, set[str]]:
    """Collect the citing-node set for each root paper."""
    root_citers = {}
    for record in records:
        root_citers[record["root_id"]] = set(graph.predecessors(record["node_id"]))
    return root_citers


def co_citation_cosine(citers_a: set[str], citers_b: set[str]) -> float:
    """Compute Salton cosine similarity from shared citing nodes."""
    if not citers_a or not citers_b:
        return 0.0
    return len(citers_a & citers_b) / math.sqrt(len(citers_a) * len(citers_b))


def co_citation_jaccard(citers_a: set[str], citers_b: set[str]) -> float:
    """Compute Jaccard similarity from shared citing nodes."""
    if not citers_a and not citers_b:
        return 0.0
    union_size = len(citers_a | citers_b)
    if union_size == 0:
        return 0.0
    return len(citers_a & citers_b) / union_size


def build_cocitation_outputs(
    records: list[dict],
    root_citers: dict[str, set[str]],
):
    """Build root table, similarity matrix, and pairwise co-citation outputs."""
    root_ids = [record["root_id"] for record in records]
    node_by_root_id = {record["root_id"]: record["node_id"] for record in records}
    title_by_root_id = {record["root_id"]: record["title"] for record in records}

    matrix_rows = []
    edge_rows = []
    for source_id in root_ids:
        matrix_row = {"root_id": source_id}
        for target_id in root_ids:
            cosine = co_citation_cosine(root_citers[source_id], root_citers[target_id])
            matrix_row[target_id] = round(cosine, 6)
            if source_id < target_id:
                shared_citers = root_citers[source_id] & root_citers[target_id]
                edge_rows.append(
                    {
                        "source_root_id": source_id,
                        "target_root_id": target_id,
                        "source_node_id": node_by_root_id[source_id],
                        "target_node_id": node_by_root_id[target_id],
                        "source_title": title_by_root_id[source_id],
                        "target_title": title_by_root_id[target_id],
                        "source_citer_count": len(root_citers[source_id]),
                        "target_citer_count": len(root_citers[target_id]),
                        "shared_citer_count": len(shared_citers),
                        "co_citation_cosine": round(cosine, 6),
                        "co_citation_jaccard": round(
                            co_citation_jaccard(
                                root_citers[source_id], root_citers[target_id]
                            ),
                            6,
                        ),
                    }
                )
        matrix_rows.append(matrix_row)

    root_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "tokens"}
        row["citer_count"] = len(root_citers[record["root_id"]])
        root_rows.append(row)

    root_df = pd.DataFrame(root_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    edges_df = pd.DataFrame(edge_rows).sort_values(
        ["co_citation_cosine", "shared_citer_count"],
        ascending=[False, False],
    )
    return root_df, matrix_df, edges_df


def build_similarity_graph(
    records: list[dict],
    root_citers: dict[str, set[str]],
    min_similarity: float,
) -> nx.Graph:
    """Create a weighted root-only graph from co-citation cosine similarity."""
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
            cosine = co_citation_cosine(root_citers[source_id], root_citers[target_id])
            if cosine >= min_similarity:
                graph.add_edge(source_id, target_id, weight=cosine)
    return graph


def summarize_shared_citers(
    community: set[str],
    root_citers: dict[str, set[str]],
    top_n: int = 8,
) -> list[str]:
    """Summarize a co-citation community by its most common citing nodes."""
    counter: dict[str, int] = {}
    for root_id in community:
        for citer in root_citers[root_id]:
            counter[citer] = counter.get(citer, 0) + 1
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [node_id for node_id, _ in ordered[:top_n]]


def detect_subtopics(
    records: list[dict],
    root_citers: dict[str, set[str]],
    min_similarity: float,
):
    """Group root papers into co-citation-based subtopics."""
    similarity_graph = build_similarity_graph(records, root_citers, min_similarity)
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
        shared_citers = summarize_shared_citers(community, root_citers)
        subtopic_label = "; ".join(shared_citers[:3]) if shared_citers else "isolated_root"

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
                "top_shared_citers": "; ".join(shared_citers),
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
                    "citer_count": len(root_citers[root_id]),
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
        description="Compute co-citation similarity and subtopics among ToS root papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.10,
        help="Minimum co-citation cosine to connect two roots in the subtopic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    root_citers = compute_root_citers(graph, records)
    root_df, matrix_df, edges_df = build_cocitation_outputs(records, root_citers)
    similarity_graph, subtopics_df, summary_df = detect_subtopics(
        records,
        root_citers,
        args.min_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_df.to_csv(args.output_dir / "root_papers.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_cocitation_matrix.csv", index=False)
    edges_df.to_csv(args.output_dir / "root_cocitation_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_cocitation_subtopics.csv", index=False)
    summary_df.to_csv(
        args.output_dir / "root_cocitation_subtopic_summary.csv",
        index=False,
    )

    print(f"Root papers: {len(root_df):,}")
    print(f"Co-citation pairs: {len(edges_df):,}")
    print(f"Subtopics: {len(summary_df):,}")
    print(
        "Subtopic similarity edges: "
        f"{similarity_graph.number_of_edges():,} (threshold={args.min_similarity:.2f})"
    )
    print(f"Saved root papers: {(args.output_dir / 'root_papers.csv').resolve()}")
    print(
        "Saved co-citation matrix: "
        f"{(args.output_dir / 'root_cocitation_matrix.csv').resolve()}"
    )
    print(
        "Saved co-citation pairs: "
        f"{(args.output_dir / 'root_cocitation_pairs.csv').resolve()}"
    )
    print(
        "Saved co-citation subtopics: "
        f"{(args.output_dir / 'root_cocitation_subtopics.csv').resolve()}"
    )
    print(
        "Saved co-citation subtopic summary: "
        f"{(args.output_dir / 'root_cocitation_subtopic_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
