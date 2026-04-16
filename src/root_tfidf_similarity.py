from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

from build_citation_network import (
    apply_sap_baseline,
    build_citation_network,
    enrich_network_with_article_data,
    load_article_metadata,
    load_citation_edges,
    remove_self_loops,
)
from sap import Sap

DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/root_tfidf")
TOKEN_PATTERN = re.compile(r"[a-z][a-z]+")
STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "also",
    "among",
    "analysis",
    "and",
    "are",
    "article",
    "based",
    "been",
    "between",
    "both",
    "but",
    "can",
    "case",
    "data",
    "development",
    "during",
    "each",
    "effect",
    "evidence",
    "findings",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "not",
    "our",
    "paper",
    "research",
    "result",
    "results",
    "show",
    "study",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "using",
    "was",
    "were",
    "which",
    "while",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize title and abstract text for a small root-paper corpus."""
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def build_sap_graph(citation_csv: Path, articles_csv: Path):
    """Rebuild the current SAP-enriched graph from BibFusion outputs."""
    citation_df = load_citation_edges(citation_csv)
    article_df = load_article_metadata(articles_csv)
    graph = build_citation_network(citation_df)
    graph = remove_self_loops(graph)
    graph = Sap.clean_graph(graph)
    graph = enrich_network_with_article_data(graph, article_df)
    return apply_sap_baseline(graph)


def extract_root_records(graph) -> list[dict]:
    """Extract root nodes and their text fields."""
    records = []
    roots = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if attrs.get("ToS") == "root"
    ]
    roots = sorted(roots, key=lambda item: item[1].get("sap_rank", 0), reverse=True)

    for index, (node, attrs) in enumerate(roots, start=1):
        title = attrs.get("title") or ""
        abstract = attrs.get("abstract") or ""
        text = f"{title} {abstract}".strip()
        tokens = tokenize(text)
        records.append(
            {
                "root_id": f"root_{index:02d}",
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
    """Compute TF-IDF vectors without external dependencies."""
    document_tokens = {record["root_id"]: record["tokens"] for record in records}
    document_count = len(document_tokens)
    document_frequency = Counter()

    for tokens in document_tokens.values():
        document_frequency.update(set(tokens))

    vectors = {}
    for root_id, tokens in document_tokens.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        vector = {}
        if total_terms == 0:
            vectors[root_id] = vector
            continue

        for term, count in term_counts.items():
            term_frequency = count / total_terms
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[term])
            ) + 1
            vector[term] = term_frequency * inverse_document_frequency
        vectors[root_id] = vector

    return vectors


def cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vector_a or not vector_b:
        return 0.0

    shared_terms = set(vector_a) & set(vector_b)
    dot_product = sum(vector_a[term] * vector_b[term] for term in shared_terms)
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def build_similarity_outputs(records: list[dict], vectors: dict[str, dict[str, float]]):
    """Build root paper, matrix, and pairwise edge outputs."""
    root_ids = [record["root_id"] for record in records]
    node_by_root_id = {record["root_id"]: record["node_id"] for record in records}
    title_by_root_id = {record["root_id"]: record["title"] for record in records}

    matrix_rows = []
    edge_rows = []
    for source_id in root_ids:
        matrix_row = {"root_id": source_id}
        for target_id in root_ids:
            similarity = cosine_similarity(vectors[source_id], vectors[target_id])
            matrix_row[target_id] = round(similarity, 6)
            if source_id < target_id:
                edge_rows.append(
                    {
                        "source_root_id": source_id,
                        "target_root_id": target_id,
                        "source_node_id": node_by_root_id[source_id],
                        "target_node_id": node_by_root_id[target_id],
                        "source_title": title_by_root_id[source_id],
                        "target_title": title_by_root_id[target_id],
                        "tfidf_similarity": round(similarity, 6),
                    }
                )
        matrix_rows.append(matrix_row)

    root_rows = []
    for record in records:
        vector = vectors[record["root_id"]]
        top_terms = sorted(vector.items(), key=lambda item: item[1], reverse=True)[:10]
        row = {
            key: value
            for key, value in record.items()
            if key not in {"tokens"}
        }
        row["top_tfidf_terms"] = "; ".join(term for term, _ in top_terms)
        root_rows.append(row)

    root_df = pd.DataFrame(root_rows)
    matrix_df = pd.DataFrame(matrix_rows)
    edges_df = pd.DataFrame(edge_rows).sort_values(
        "tfidf_similarity", ascending=False
    )
    return root_df, matrix_df, edges_df


def build_similarity_graph(
    records: list[dict],
    vectors: dict[str, dict[str, float]],
    min_similarity: float,
) -> nx.Graph:
    """Create a weighted root-only graph from TF-IDF cosine similarity."""
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
            similarity = cosine_similarity(vectors[source_id], vectors[target_id])
            if similarity >= min_similarity:
                graph.add_edge(source_id, target_id, weight=similarity)
    return graph


def aggregate_top_terms(
    community: set[str],
    vectors: dict[str, dict[str, float]],
    top_n: int = 8,
) -> list[str]:
    """Summarize a TF-IDF community with its strongest aggregate terms."""
    aggregate_scores = Counter()
    for root_id in community:
        aggregate_scores.update(vectors[root_id])
    return [term for term, _ in aggregate_scores.most_common(top_n)]


def detect_subtopics(
    records: list[dict],
    vectors: dict[str, dict[str, float]],
    min_similarity: float,
):
    """Group root papers into TF-IDF-based subtopics."""
    similarity_graph = build_similarity_graph(records, vectors, min_similarity)
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
        top_terms = aggregate_top_terms(community, vectors)
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
        description="Compute TF-IDF similarity and subtopics among ToS root papers."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.08,
        help="Minimum cosine similarity to connect two roots in the subtopic graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    records = extract_root_records(graph)
    vectors = compute_tfidf_vectors(records)
    root_df, matrix_df, edges_df = build_similarity_outputs(records, vectors)
    (
        subtopic_graph,
        subtopics_df,
        summary_df,
    ) = detect_subtopics(records, vectors, args.min_similarity)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root_df.to_csv(args.output_dir / "root_papers.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_similarity_matrix.csv", index=False)
    edges_df.to_csv(args.output_dir / "root_similarity_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_subtopics.csv", index=False)
    summary_df.to_csv(args.output_dir / "root_subtopic_summary.csv", index=False)

    print(f"Root papers: {len(root_df):,}")
    print(f"Similarity pairs: {len(edges_df):,}")
    print(f"Subtopics: {len(summary_df):,}")
    print(
        "Subtopic similarity edges: "
        f"{subtopic_graph.number_of_edges():,} (threshold={args.min_similarity:.2f})"
    )
    print(f"Saved root papers: {(args.output_dir / 'root_papers.csv').resolve()}")
    print(
        "Saved similarity matrix: "
        f"{(args.output_dir / 'root_similarity_matrix.csv').resolve()}"
    )
    print(
        "Saved similarity pairs: "
        f"{(args.output_dir / 'root_similarity_pairs.csv').resolve()}"
    )
    print(
        "Saved root subtopics: "
        f"{(args.output_dir / 'root_subtopics.csv').resolve()}"
    )
    print(
        "Saved subtopic summary: "
        f"{(args.output_dir / 'root_subtopic_summary.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
