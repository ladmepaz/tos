from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from root_cocitation_similarity import co_citation_cosine, compute_root_citers
from root_structural_similarity import compute_root_influence_sets, overlap_cosine
from root_tfidf_similarity import (
    build_sap_graph,
    compute_tfidf_vectors,
    cosine_similarity,
    extract_root_records,
)

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_combined")


def aggregate_top_terms(
    community: set[str],
    vectors: dict[str, dict[str, float]],
    top_n: int = 8,
) -> list[str]:
    """Summarize a combined-similarity community with TF-IDF terms."""
    aggregate_scores = Counter()
    for root_id in community:
        aggregate_scores.update(vectors[root_id])
    return [term for term, _ in aggregate_scores.most_common(top_n)]


def build_combined_outputs(
    records: list[dict],
    tfidf_vectors: dict[str, dict[str, float]],
    root_citers: dict[str, set[str]],
    influence_sets: dict[str, set[str]],
    w_text: float,
    w_cocitation: float,
    w_structural: float,
):
    """Build root table, combined matrix, and pairwise outputs."""
    root_ids = [record["root_id"] for record in records]
    node_by_root_id = {record["root_id"]: record["node_id"] for record in records}
    title_by_root_id = {record["root_id"]: record["title"] for record in records}

    matrix_rows = []
    edge_rows = []
    for source_id in root_ids:
        matrix_row = {"root_id": source_id}
        for target_id in root_ids:
            text_similarity = cosine_similarity(
                tfidf_vectors[source_id], tfidf_vectors[target_id]
            )
            co_citation_similarity = co_citation_cosine(
                root_citers[source_id], root_citers[target_id]
            )
            structural_similarity = overlap_cosine(
                influence_sets[source_id], influence_sets[target_id]
            )
            combined_similarity = (
                (w_text * text_similarity)
                + (w_cocitation * co_citation_similarity)
                + (w_structural * structural_similarity)
            )
            matrix_row[target_id] = round(combined_similarity, 6)
            if source_id < target_id:
                edge_rows.append(
                    {
                        "source_root_id": source_id,
                        "target_root_id": target_id,
                        "source_node_id": node_by_root_id[source_id],
                        "target_node_id": node_by_root_id[target_id],
                        "source_title": title_by_root_id[source_id],
                        "target_title": title_by_root_id[target_id],
                        "text_similarity": round(text_similarity, 6),
                        "co_citation_similarity": round(co_citation_similarity, 6),
                        "structural_similarity": round(structural_similarity, 6),
                        "combined_similarity": round(combined_similarity, 6),
                    }
                )
        matrix_rows.append(matrix_row)

    root_rows = []
    for record in records:
        vector = tfidf_vectors[record["root_id"]]
        top_terms = sorted(vector.items(), key=lambda item: item[1], reverse=True)[:10]
        row = {key: value for key, value in record.items() if key != "tokens"}
        row["citer_count"] = len(root_citers[record["root_id"]])
        row["influence_count"] = len(influence_sets[record["root_id"]])
        row["top_tfidf_terms"] = "; ".join(term for term, _ in top_terms)
        root_rows.append(row)

    root_df = pd.DataFrame(root_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    edges_df = pd.DataFrame(edge_rows).sort_values(
        "combined_similarity",
        ascending=False,
    )
    return root_df, matrix_df, edges_df


def build_similarity_graph(
    records: list[dict],
    edges_df: pd.DataFrame,
    min_similarity: float,
) -> nx.Graph:
    """Create a weighted root-only graph from combined similarity."""
    graph = nx.Graph()
    for record in records:
        graph.add_node(
            record["root_id"],
            node_id=record["node_id"],
            sap_rank=record["sap_rank"],
            title=record["title"],
        )

    for row in edges_df.itertuples(index=False):
        if row.combined_similarity >= min_similarity:
            graph.add_edge(
                row.source_root_id,
                row.target_root_id,
                weight=row.combined_similarity,
            )
    return graph


def detect_subtopics(
    records: list[dict],
    tfidf_vectors: dict[str, dict[str, float]],
    edges_df: pd.DataFrame,
    min_similarity: float,
):
    """Group root papers into combined-similarity subtopics."""
    similarity_graph = build_similarity_graph(records, edges_df, min_similarity)
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
        top_terms = aggregate_top_terms(community, tfidf_vectors)
        subtopic_label = "; ".join(top_terms[:5])
        representative_root_id = max(
            community,
            key=lambda root_id: root_lookup[root_id]["sap_rank"],
        )
        representative_node_id = root_lookup[representative_root_id]["node_id"]

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
                "top_terms": "; ".join(top_terms),
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
        description="Compute combined text/co-citation/structural similarity among ToS root papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-text", type=float, default=0.45)
    parser.add_argument("--w-cocitation", type=float, default=0.35)
    parser.add_argument("--w-structural", type=float, default=0.20)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="Minimum combined similarity to connect two roots in the subtopic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight_sum = args.w_text + args.w_cocitation + args.w_structural
    if weight_sum <= 0:
        raise ValueError("At least one weight must be positive.")

    w_text = args.w_text / weight_sum
    w_cocitation = args.w_cocitation / weight_sum
    w_structural = args.w_structural / weight_sum

    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    tfidf_vectors = compute_tfidf_vectors(records)
    root_citers = compute_root_citers(graph, records)
    influence_sets = compute_root_influence_sets(graph, records)
    root_df, matrix_df, edges_df = build_combined_outputs(
        records,
        tfidf_vectors,
        root_citers,
        influence_sets,
        w_text,
        w_cocitation,
        w_structural,
    )
    similarity_graph, subtopics_df, summary_df = detect_subtopics(
        records,
        tfidf_vectors,
        edges_df,
        args.min_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_df.to_csv(args.output_dir / "root_papers.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_combined_matrix.csv", index=False)
    edges_df.to_csv(args.output_dir / "root_combined_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_combined_subtopics.csv", index=False)
    summary_df.to_csv(args.output_dir / "root_combined_subtopic_summary.csv", index=False)

    print(f"Root papers: {len(root_df):,}")
    print(f"Combined pairs: {len(edges_df):,}")
    print(f"Subtopics: {len(summary_df):,}")
    print(
        "Subtopic similarity edges: "
        f"{similarity_graph.number_of_edges():,} (threshold={args.min_similarity:.2f})"
    )
    print(
        "Weights used: "
        f"text={w_text:.2f}, co_citation={w_cocitation:.2f}, structural={w_structural:.2f}"
    )
    print(f"Saved root papers: {(args.output_dir / 'root_papers.csv').resolve()}")
    print(
        "Saved combined matrix: "
        f"{(args.output_dir / 'root_combined_matrix.csv').resolve()}"
    )
    print(
        "Saved combined pairs: "
        f"{(args.output_dir / 'root_combined_pairs.csv').resolve()}"
    )
    print(
        "Saved combined subtopics: "
        f"{(args.output_dir / 'root_combined_subtopics.csv').resolve()}"
    )
    print(
        "Saved combined subtopic summary: "
        f"{(args.output_dir / 'root_combined_subtopic_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
