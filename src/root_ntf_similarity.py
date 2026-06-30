from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import networkx as nx


DEFAULT_INPUT = Path("outputs/root_tfidf/root_papers.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/root_ntf")
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
    """Tokenize root title and abstract text for the exploratory NTF run."""
    return [
        token
        for token in TOKEN_PATTERN.findall(str(text).lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def load_root_documents(input_csv: Path) -> list[dict]:
    root_df = pd.read_csv(input_csv)
    records = []
    for row in root_df.itertuples(index=False):
        title = "" if pd.isna(row.title) else str(row.title)
        abstract = "" if pd.isna(row.abstract) else str(row.abstract)
        text = f"{title} {abstract}".strip()
        records.append(
            {
                "root_id": row.root_id,
                "node_id": row.node_id,
                "sap_rank": row.sap_rank,
                "title": title,
                "abstract": abstract,
                "year": row.year,
                "journal": "" if pd.isna(row.journal) else str(row.journal),
                "tokens": tokenize(text),
            }
        )
    return records


def build_vocabulary(records: list[dict], min_df: int, max_df: float) -> list[str]:
    doc_count = len(records)
    document_frequency = Counter()
    for record in records:
        document_frequency.update(set(record["tokens"]))

    max_df_count = max_df * doc_count if max_df <= 1 else max_df
    terms = [
        term
        for term, count in document_frequency.items()
        if count >= min_df and count <= max_df_count
    ]
    return sorted(terms)


def build_count_matrix(records: list[dict], terms: list[str]) -> np.ndarray:
    term_index = {term: index for index, term in enumerate(terms)}
    matrix = np.zeros((len(records), len(terms)), dtype=float)
    for doc_index, record in enumerate(records):
        counts = Counter(record["tokens"])
        for term, count in counts.items():
            if term in term_index:
                matrix[doc_index, term_index[term]] = count
    return matrix


def tfidf_weight(count_matrix: np.ndarray) -> np.ndarray:
    doc_count = count_matrix.shape[0]
    document_frequency = (count_matrix > 0).sum(axis=0)
    inverse_document_frequency = np.log((1 + doc_count) / (1 + document_frequency)) + 1
    return np.log1p(count_matrix) * inverse_document_frequency


def compute_term_entropy(count_matrix: np.ndarray) -> np.ndarray:
    doc_count = count_matrix.shape[0]
    global_frequency = count_matrix.sum(axis=0)
    safe_global_frequency = np.where(global_frequency > 0, global_frequency, 1.0)
    probabilities = count_matrix / safe_global_frequency
    with np.errstate(divide="ignore", invalid="ignore"):
        p_log_p = np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0)
    return -p_log_p.sum(axis=0) / math.log(doc_count)


def build_entropy_tensor(
    count_matrix: np.ndarray,
    weighted_matrix: np.ndarray,
    entropy: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(entropy, np.linspace(0, 1, bins + 1))
    term_bins = np.digitize(entropy, edges[1:-1])
    tensor = np.zeros((weighted_matrix.shape[0], weighted_matrix.shape[1], bins), dtype=float)
    for bin_index in range(bins):
        mask = term_bins == bin_index
        tensor[:, mask, bin_index] = weighted_matrix[:, mask]
    return tensor, term_bins


def mttkrp(tensor: np.ndarray, factors: tuple[np.ndarray, np.ndarray, np.ndarray], mode: int) -> np.ndarray:
    doc_factor, term_factor, bin_factor = factors
    if mode == 0:
        return np.einsum("ijk,jr,kr->ir", tensor, term_factor, bin_factor)
    if mode == 1:
        return np.einsum("ijk,ir,kr->jr", tensor, doc_factor, bin_factor)
    if mode == 2:
        return np.einsum("ijk,ir,jr->kr", tensor, doc_factor, term_factor)
    raise ValueError("mode must be 0, 1, or 2")


def nonnegative_cp(
    tensor: np.ndarray,
    rank: int,
    iterations: int,
    random_state: int,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    doc_factor = rng.random((tensor.shape[0], rank)) + 0.1
    term_factor = rng.random((tensor.shape[1], rank)) + 0.1
    bin_factor = rng.random((tensor.shape[2], rank)) + 0.1

    for _ in range(iterations):
        numerator = mttkrp(tensor, (doc_factor, term_factor, bin_factor), mode=0)
        denominator = doc_factor @ ((term_factor.T @ term_factor) * (bin_factor.T @ bin_factor))
        doc_factor *= numerator / np.maximum(denominator, epsilon)

        numerator = mttkrp(tensor, (doc_factor, term_factor, bin_factor), mode=1)
        denominator = term_factor @ ((doc_factor.T @ doc_factor) * (bin_factor.T @ bin_factor))
        term_factor *= numerator / np.maximum(denominator, epsilon)

        numerator = mttkrp(tensor, (doc_factor, term_factor, bin_factor), mode=2)
        denominator = bin_factor @ ((doc_factor.T @ doc_factor) * (term_factor.T @ term_factor))
        bin_factor *= numerator / np.maximum(denominator, epsilon)

        norms = np.maximum(np.linalg.norm(term_factor, axis=0), epsilon)
        term_factor /= norms
        doc_factor *= norms

    return doc_factor, term_factor, bin_factor


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / np.maximum(row_sums, 1e-10)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-10)
    return normalized @ normalized.T


def build_outputs(
    records: list[dict],
    terms: list[str],
    doc_topics: np.ndarray,
    term_factor: np.ndarray,
    similarity_matrix: np.ndarray,
    top_terms: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topic_rows = []
    for topic_index in range(term_factor.shape[1]):
        weights = term_factor[:, topic_index]
        top_indices = weights.argsort()[-top_terms:][::-1]
        topic_rows.append(
            {
                "ntf_topic": topic_index,
                "top_terms": "; ".join(terms[index] for index in top_indices),
                "top_weights": "; ".join(f"{weights[index]:.6f}" for index in top_indices),
            }
        )

    root_rows = []
    for record, topic_weights in zip(records, doc_topics):
        dominant_topic = int(np.argmax(topic_weights))
        root_rows.append(
            {
                "root_id": record["root_id"],
                "node_id": record["node_id"],
                "title": record["title"],
                "dominant_ntf_topic": dominant_topic,
                "ntf_topic_confidence": float(topic_weights[dominant_topic]),
                **{
                    f"ntf_topic_{topic_index}": float(value)
                    for topic_index, value in enumerate(topic_weights)
                },
            }
        )

    root_ids = [record["root_id"] for record in records]
    matrix_rows = []
    pair_rows = []
    for source_index, source_record in enumerate(records):
        row = {"root_id": source_record["root_id"]}
        for target_index, target_record in enumerate(records):
            similarity = float(similarity_matrix[source_index, target_index])
            row[target_record["root_id"]] = round(similarity, 6)
            if source_index < target_index:
                pair_rows.append(
                    {
                        "source_root_id": source_record["root_id"],
                        "target_root_id": target_record["root_id"],
                        "source_node_id": source_record["node_id"],
                        "target_node_id": target_record["node_id"],
                        "source_title": source_record["title"],
                        "target_title": target_record["title"],
                        "ntf_topic_similarity": round(similarity, 6),
                    }
                )
        matrix_rows.append(row)

    return (
        pd.DataFrame(topic_rows),
        pd.DataFrame(root_rows),
        pd.DataFrame(matrix_rows, columns=["root_id", *root_ids]),
        pd.DataFrame(pair_rows).sort_values("ntf_topic_similarity", ascending=False),
    )


def detect_ntf_subtopics(
    roots_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    min_similarity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect exploratory root groups from the NTF topic-similarity graph."""
    graph = nx.Graph()
    for row in roots_df.itertuples(index=False):
        graph.add_node(
            row.root_id,
            node_id=row.node_id,
            title=row.title,
            dominant_ntf_topic=int(row.dominant_ntf_topic),
            ntf_topic_confidence=float(row.ntf_topic_confidence),
        )

    for row in pairs_df.itertuples(index=False):
        if row.ntf_topic_similarity >= min_similarity:
            graph.add_edge(
                row.source_root_id,
                row.target_root_id,
                weight=float(row.ntf_topic_similarity),
            )

    if graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes]
    else:
        communities = list(
            nx.community.greedy_modularity_communities(graph, weight="weight")
        )
        assigned_nodes = set().union(*communities) if communities else set()
        communities.extend({node} for node in set(graph.nodes) - assigned_nodes)

    root_lookup = roots_df.set_index("root_id").to_dict(orient="index")
    communities = sorted(
        communities,
        key=lambda community: (
            -len(community),
            -max(root_lookup[root_id]["ntf_topic_confidence"] for root_id in community),
        ),
    )

    subtopic_rows = []
    summary_rows = []
    for index, community in enumerate(communities, start=1):
        subtopic_id = f"ntf_subtopic_{index}"
        member_root_ids = sorted(
            community,
            key=lambda root_id: root_lookup[root_id]["ntf_topic_confidence"],
            reverse=True,
        )
        member_topics = Counter(
            int(root_lookup[root_id]["dominant_ntf_topic"]) for root_id in member_root_ids
        )
        dominant_topic = member_topics.most_common(1)[0][0]
        member_node_ids = [root_lookup[root_id]["node_id"] for root_id in member_root_ids]
        representative_root_id = member_root_ids[0]

        summary_rows.append(
            {
                "ntf_subtopic_id": subtopic_id,
                "member_count": len(member_root_ids),
                "dominant_ntf_topic": dominant_topic,
                "representative_root_id": representative_root_id,
                "representative_node_id": root_lookup[representative_root_id]["node_id"],
                "member_root_ids": "; ".join(member_root_ids),
                "member_node_ids": "; ".join(member_node_ids),
            }
        )

        for root_id in member_root_ids:
            root_data = root_lookup[root_id]
            subtopic_rows.append(
                {
                    "ntf_subtopic_id": subtopic_id,
                    "root_id": root_id,
                    "node_id": root_data["node_id"],
                    "title": root_data["title"],
                    "dominant_ntf_topic": int(root_data["dominant_ntf_topic"]),
                    "ntf_topic_confidence": float(root_data["ntf_topic_confidence"]),
                }
            )

    return (
        pd.DataFrame(subtopic_rows),
        pd.DataFrame(summary_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exploratory root-only NTF topic similarity."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.85)
    parser.add_argument("--top-terms", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-similarity", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_root_documents(args.input)
    terms = build_vocabulary(records, args.min_df, args.max_df)
    if not terms:
        raise ValueError("No terms available after min_df/max_df filtering.")

    count_matrix = build_count_matrix(records, terms)
    weighted_matrix = tfidf_weight(count_matrix)
    entropy = compute_term_entropy(count_matrix)
    tensor, _ = build_entropy_tensor(count_matrix, weighted_matrix, entropy, args.bins)
    doc_factor, term_factor, _ = nonnegative_cp(
        tensor,
        rank=args.rank,
        iterations=args.iterations,
        random_state=args.random_state,
    )
    doc_topics = row_normalize(doc_factor)
    similarity_matrix = cosine_similarity_matrix(doc_topics)
    topics_df, roots_df, matrix_df, pairs_df = build_outputs(
        records,
        terms,
        doc_topics,
        term_factor,
        similarity_matrix,
        args.top_terms,
    )
    subtopics_df, summary_df = detect_ntf_subtopics(
        roots_df,
        pairs_df,
        args.min_similarity,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(args.output_dir / "root_ntf_topics.csv", index=False)
    roots_df.to_csv(args.output_dir / "root_ntf_topic_profiles.csv", index=False)
    matrix_df.to_csv(args.output_dir / "root_ntf_similarity_matrix.csv", index=False)
    pairs_df.to_csv(args.output_dir / "root_ntf_similarity_pairs.csv", index=False)
    subtopics_df.to_csv(args.output_dir / "root_ntf_subtopics.csv", index=False)
    summary_df.to_csv(args.output_dir / "root_ntf_subtopic_summary.csv", index=False)

    print(f"Root papers: {len(records):,}")
    print(f"Terms: {len(terms):,}")
    print(f"Rank: {args.rank:,}")
    print(f"Bins: {args.bins:,}")
    print(f"NTF pairs: {len(pairs_df):,}")
    print(f"NTF subtopics: {len(summary_df):,} (threshold={args.min_similarity:.2f})")
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
