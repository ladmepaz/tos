from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from root_combined_similarity import normalize_weights
from root_tfidf_similarity import build_sap_graph, cosine_similarity, tokenize


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/trunk_combined")


def extract_trunk_records(graph: nx.DiGraph) -> list[dict]:
    """Extract trunk nodes and their text fields from the SAP graph."""
    trunk_nodes = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if attrs.get("ToS") == "trunk"
    ]
    trunk_nodes = sorted(
        trunk_nodes,
        key=lambda item: item[1].get("sap_rank", 0),
        reverse=True,
    )

    records = []
    for index, (node, attrs) in enumerate(trunk_nodes, start=1):
        title = attrs.get("title") or ""
        abstract = attrs.get("abstract") or ""
        text = f"{title} {abstract}".strip()
        tokens = tokenize(text)
        records.append(
            {
                "trunk_id": f"trunk_{index:02d}",
                "node_id": node,
                "sap_rank": attrs.get("sap_rank", 0),
                "title": title,
                "abstract": abstract,
                "year": attrs.get("year") or "",
                "journal": attrs.get("journal") or "",
                "source_title": attrs.get("source_title") or "",
                "doi": attrs.get("doi") or "",
                "has_article_metadata": attrs.get("has_article_metadata", False),
                "token_count": len(tokens),
                "tokens": tokens,
            }
        )
    return records


def compute_tfidf_vectors(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute TF-IDF vectors for trunk records."""
    document_tokens = {record["trunk_id"]: record["tokens"] for record in records}
    document_count = len(document_tokens)
    document_frequency = Counter()

    for tokens in document_tokens.values():
        document_frequency.update(set(tokens))

    vectors = {}
    for trunk_id, tokens in document_tokens.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        vector = {}
        if total_terms == 0:
            vectors[trunk_id] = vector
            continue

        for term, count in term_counts.items():
            term_frequency = count / total_terms
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[term])
            ) + 1
            vector[term] = term_frequency * inverse_document_frequency
        vectors[trunk_id] = vector

    return vectors


def overlap_cosine(items_a: set[str], items_b: set[str]) -> float:
    """Compute Salton cosine similarity for two node sets."""
    if not items_a or not items_b:
        return 0.0
    return len(items_a & items_b) / math.sqrt(len(items_a) * len(items_b))


def compute_trunk_reference_sets(
    graph: nx.DiGraph,
    records: list[dict],
) -> dict[str, set[str]]:
    """Collect papers cited by each trunk node."""
    return {
        record["trunk_id"]: set(graph.successors(record["node_id"]))
        for record in records
    }


def compute_trunk_citer_sets(
    graph: nx.DiGraph,
    records: list[dict],
) -> dict[str, set[str]]:
    """Collect papers citing each trunk node."""
    return {
        record["trunk_id"]: set(graph.predecessors(record["node_id"]))
        for record in records
    }


def aggregate_top_terms(
    community: set[str],
    vectors: dict[str, dict[str, float]],
    top_n: int = 8,
) -> list[str]:
    """Summarize a trunk subgroup with aggregate TF-IDF terms."""
    aggregate_scores = Counter()
    for trunk_id in community:
        aggregate_scores.update(vectors[trunk_id])
    return [term for term, _ in aggregate_scores.most_common(top_n)]


def build_combined_outputs(
    records: list[dict],
    tfidf_vectors: dict[str, dict[str, float]],
    reference_sets: dict[str, set[str]],
    citer_sets: dict[str, set[str]],
    w_text: float,
    w_references: float,
    w_citers: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build trunk table, combined matrix, and pairwise similarity outputs."""
    trunk_ids = [record["trunk_id"] for record in records]
    node_by_trunk_id = {record["trunk_id"]: record["node_id"] for record in records}
    title_by_trunk_id = {record["trunk_id"]: record["title"] for record in records}

    matrix_rows = []
    pair_rows = []
    for source_id in trunk_ids:
        matrix_row = {"trunk_id": source_id}
        for target_id in trunk_ids:
            text_similarity = cosine_similarity(
                tfidf_vectors[source_id],
                tfidf_vectors[target_id],
            )
            shared_reference_similarity = overlap_cosine(
                reference_sets[source_id],
                reference_sets[target_id],
            )
            shared_citer_similarity = overlap_cosine(
                citer_sets[source_id],
                citer_sets[target_id],
            )
            combined_similarity = (
                (w_text * text_similarity)
                + (w_references * shared_reference_similarity)
                + (w_citers * shared_citer_similarity)
            )
            matrix_row[target_id] = round(combined_similarity, 6)
            if source_id < target_id:
                pair_rows.append(
                    {
                        "source_trunk_id": source_id,
                        "target_trunk_id": target_id,
                        "source_node_id": node_by_trunk_id[source_id],
                        "target_node_id": node_by_trunk_id[target_id],
                        "source_title": title_by_trunk_id[source_id],
                        "target_title": title_by_trunk_id[target_id],
                        "text_similarity": round(text_similarity, 6),
                        "shared_reference_similarity": round(
                            shared_reference_similarity,
                            6,
                        ),
                        "shared_citer_similarity": round(shared_citer_similarity, 6),
                        "combined_similarity": round(combined_similarity, 6),
                    }
                )
        matrix_rows.append(matrix_row)

    trunk_rows = []
    for record in records:
        vector = tfidf_vectors[record["trunk_id"]]
        top_terms = sorted(vector.items(), key=lambda item: item[1], reverse=True)[:10]
        row = {key: value for key, value in record.items() if key != "tokens"}
        row["reference_count"] = len(reference_sets[record["trunk_id"]])
        row["citer_count"] = len(citer_sets[record["trunk_id"]])
        row["top_tfidf_terms"] = "; ".join(term for term, _ in top_terms)
        trunk_rows.append(row)

    trunk_df = pd.DataFrame(trunk_rows)
    matrix_df = pd.DataFrame(matrix_rows, columns=["trunk_id", *trunk_ids])
    pairs_df = pd.DataFrame(pair_rows).sort_values(
        "combined_similarity",
        ascending=False,
    )
    return trunk_df, matrix_df, pairs_df


def build_similarity_graph(
    records: list[dict],
    pairs_df: pd.DataFrame,
    min_similarity: float,
) -> nx.Graph:
    """Create a weighted trunk-only graph from combined similarity."""
    graph = nx.Graph()
    for record in records:
        graph.add_node(
            record["trunk_id"],
            node_id=record["node_id"],
            sap_rank=record["sap_rank"],
            title=record["title"],
        )

    for row in pairs_df.itertuples(index=False):
        if row.combined_similarity >= min_similarity:
            graph.add_edge(
                row.source_trunk_id,
                row.target_trunk_id,
                weight=row.combined_similarity,
            )
    return graph


def detect_subtopics(
    records: list[dict],
    tfidf_vectors: dict[str, dict[str, float]],
    pairs_df: pd.DataFrame,
    min_similarity: float,
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """Group trunk nodes into preliminary subtopics."""
    similarity_graph = build_similarity_graph(records, pairs_df, min_similarity)
    if similarity_graph.number_of_edges() == 0:
        communities = [{record["trunk_id"]} for record in records]
    else:
        communities = list(
            nx.community.greedy_modularity_communities(
                similarity_graph,
                weight="weight",
            )
        )
        assigned_nodes = set().union(*communities) if communities else set()
        communities.extend({node} for node in set(similarity_graph.nodes) - assigned_nodes)

    trunk_lookup = {record["trunk_id"]: record for record in records}
    sorted_communities = sorted(
        communities,
        key=lambda community: (
            -len(community),
            -max(trunk_lookup[trunk_id]["sap_rank"] for trunk_id in community),
        ),
    )

    subtopic_rows = []
    summary_rows = []
    for index, community in enumerate(sorted_communities, start=1):
        subtopic_id = f"trunk_subtopic_{index}"
        top_terms = aggregate_top_terms(community, tfidf_vectors)
        subtopic_label = "; ".join(top_terms[:5])
        representative_trunk_id = max(
            community,
            key=lambda trunk_id: trunk_lookup[trunk_id]["sap_rank"],
        )
        representative_node_id = trunk_lookup[representative_trunk_id]["node_id"]
        member_trunk_ids = sorted(
            community,
            key=lambda trunk_id: trunk_lookup[trunk_id]["sap_rank"],
            reverse=True,
        )
        member_node_ids = [
            trunk_lookup[trunk_id]["node_id"] for trunk_id in member_trunk_ids
        ]

        summary_rows.append(
            {
                "trunk_subtopic_id": subtopic_id,
                "trunk_subtopic_label": subtopic_label,
                "member_count": len(community),
                "representative_trunk_id": representative_trunk_id,
                "representative_node_id": representative_node_id,
                "top_terms": "; ".join(top_terms),
                "member_trunk_ids": "; ".join(member_trunk_ids),
                "member_node_ids": "; ".join(member_node_ids),
            }
        )

        for trunk_id in member_trunk_ids:
            record = trunk_lookup[trunk_id]
            subtopic_rows.append(
                {
                    "trunk_subtopic_id": subtopic_id,
                    "trunk_subtopic_label": subtopic_label,
                    "is_representative": trunk_id == representative_trunk_id,
                    "trunk_id": trunk_id,
                    "node_id": record["node_id"],
                    "sap_rank": record["sap_rank"],
                    "title": record["title"],
                    "year": record["year"],
                    "journal": record["journal"],
                    "source_title": record["source_title"],
                    "has_article_metadata": record["has_article_metadata"],
                    "token_count": record["token_count"],
                }
            )

    subtopics_df = pd.DataFrame(subtopic_rows).sort_values(
        ["trunk_subtopic_id", "sap_rank"],
        ascending=[True, False],
    )
    summary_df = pd.DataFrame(summary_rows).sort_values("trunk_subtopic_id")
    return similarity_graph, subtopics_df, summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute preliminary combined similarity groups among ToS trunk papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-text", type=float, default=0.40)
    parser.add_argument("--w-references", type=float, default=0.35)
    parser.add_argument("--w-citers", type=float, default=0.25)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.18,
        help="Minimum combined similarity to connect two trunk nodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_text, w_references, w_citers = normalize_weights(
        args.w_text,
        args.w_references,
        args.w_citers,
    )

    graph = build_sap_graph(args.citations, args.articles)
    records = extract_trunk_records(graph)
    tfidf_vectors = compute_tfidf_vectors(records)
    reference_sets = compute_trunk_reference_sets(graph, records)
    citer_sets = compute_trunk_citer_sets(graph, records)
    trunk_df, matrix_df, pairs_df = build_combined_outputs(
        records,
        tfidf_vectors,
        reference_sets,
        citer_sets,
        w_text,
        w_references,
        w_citers,
    )
    similarity_graph, subtopics_df, summary_df = detect_subtopics(
        records,
        tfidf_vectors,
        pairs_df,
        args.min_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trunk_df.to_csv(args.output_dir / "trunk_papers.csv", index=False)
    matrix_df.to_csv(args.output_dir / "trunk_combined_matrix.csv", index=False)
    pairs_df.to_csv(args.output_dir / "trunk_combined_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "trunk_subtopics.csv", index=False)
    summary_df.to_csv(args.output_dir / "trunk_subtopic_summary.csv", index=False)

    print(f"Trunk papers: {len(trunk_df):,}")
    print(f"Combined pairs: {len(pairs_df):,}")
    print(f"Subtopics: {len(summary_df):,}")
    print(
        "Subtopic similarity edges: "
        f"{similarity_graph.number_of_edges():,} (threshold={args.min_similarity:.2f})"
    )
    print(
        "Weights used: "
        f"text={w_text:.2f}, references={w_references:.2f}, citers={w_citers:.2f}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
