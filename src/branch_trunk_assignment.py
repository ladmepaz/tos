from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from root_combined_similarity import normalize_weights
from root_tfidf_similarity import build_sap_graph, cosine_similarity, tokenize
from trunk_combined_similarity import overlap_cosine


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_TRUNK_SUBTOPICS = Path("outputs/trunk_combined/trunk_subtopics.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/branch_assignment")
MAIN_TRUNK_SUBTOPICS = [
    "trunk_subtopic_2",
    "trunk_subtopic_1",
    "trunk_subtopic_3",
]


def extract_branch_records(graph: nx.DiGraph) -> list[dict]:
    """Extract SAP branch nodes and their text fields."""
    branch_nodes = [
        (node, attrs)
        for node, attrs in graph.nodes(data=True)
        if str(attrs.get("ToS", "")).startswith("branch_")
    ]
    branch_nodes = sorted(
        branch_nodes,
        key=lambda item: (
            str(item[1].get("ToS", "")),
            -float(item[1].get("sap_rank", 0)),
        ),
    )

    records = []
    for index, (node, attrs) in enumerate(branch_nodes, start=1):
        title = attrs.get("title") or ""
        abstract = attrs.get("abstract") or ""
        tokens = tokenize(f"{title} {abstract}".strip())
        records.append(
            {
                "branch_id": f"branch_{index:02d}",
                "node_id": node,
                "ToS": attrs.get("ToS", ""),
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


def extract_branch_records_from_members(
    graph: nx.DiGraph,
    branch_members_csv: Path,
) -> list[dict]:
    """Extract branch records from an experimental branch-member table."""
    members_df = pd.read_csv(branch_members_csv)
    required_columns = {"branch_label", "branch_rank", "node_id"}
    missing_columns = required_columns - set(members_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing branch-member columns in {branch_members_csv}: "
            f"{sorted(missing_columns)}"
        )

    records = []
    members_df = members_df.sort_values(["branch_label", "branch_rank"])
    for row in members_df.itertuples(index=False):
        node_id = str(row.node_id)
        if node_id not in graph:
            continue
        attrs = graph.nodes[node_id]
        title = attrs.get("title") or ""
        abstract = attrs.get("abstract") or ""
        tokens = tokenize(f"{title} {abstract}".strip())
        branch_label = str(row.branch_label)
        branch_rank = int(row.branch_rank)
        records.append(
            {
                "branch_id": f"{branch_label}_{branch_rank:02d}",
                "node_id": node_id,
                "ToS": branch_label,
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


def load_main_trunk_subtopics(path: Path) -> pd.DataFrame:
    """Load the three trunk subtopics used as branch parents."""
    subtopics_df = pd.read_csv(path)
    subtopics_df = subtopics_df[
        subtopics_df["trunk_subtopic_id"].isin(MAIN_TRUNK_SUBTOPICS)
    ].copy()
    subtopics_df["trunk_subtopic_id"] = pd.Categorical(
        subtopics_df["trunk_subtopic_id"],
        categories=MAIN_TRUNK_SUBTOPICS,
        ordered=True,
    )
    return subtopics_df.sort_values(["trunk_subtopic_id", "trunk_id"])


def build_document_vectors(records: list[dict], key_field: str) -> dict[str, dict[str, float]]:
    """Compute TF-IDF vectors for a small mixed corpus."""
    document_tokens = {record[key_field]: record["tokens"] for record in records}
    document_count = len(document_tokens)
    document_frequency = Counter()

    for tokens in document_tokens.values():
        document_frequency.update(set(tokens))

    vectors = {}
    for doc_id, tokens in document_tokens.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        vector = {}
        if total_terms == 0:
            vectors[doc_id] = vector
            continue
        for term, count in term_counts.items():
            term_frequency = count / total_terms
            inverse_document_frequency = math.log(
                (1 + document_count) / (1 + document_frequency[term])
            ) + 1
            vector[term] = term_frequency * inverse_document_frequency
        vectors[doc_id] = vector
    return vectors


def build_trunk_subtopic_records(
    graph: nx.DiGraph,
    trunk_subtopics_df: pd.DataFrame,
) -> list[dict]:
    """Aggregate trunk member text into one parent record per trunk subtopic."""
    records = []
    for subtopic_id, group in trunk_subtopics_df.groupby("trunk_subtopic_id", observed=True):
        tokens: list[str] = []
        member_node_ids = group["node_id"].astype(str).tolist()
        titles = []
        for node_id in member_node_ids:
            attrs = graph.nodes[node_id]
            title = attrs.get("title") or ""
            abstract = attrs.get("abstract") or ""
            titles.append(title)
            tokens.extend(tokenize(f"{title} {abstract}".strip()))

        label = str(group["trunk_subtopic_label"].iloc[0])
        records.append(
            {
                "trunk_subtopic_id": str(subtopic_id),
                "trunk_subtopic_label": label,
                "member_count": len(member_node_ids),
                "member_node_ids": member_node_ids,
                "title": " ".join(titles),
                "tokens": tokens,
            }
        )
    return records


def citation_path_proximity(
    graph: nx.DiGraph,
    source_node: str,
    target_nodes: list[str],
    max_depth: int,
) -> tuple[float, int | None, str]:
    """Score how close a branch paper is to a trunk subtopic through citations."""
    best_distance: int | None = None
    best_target = ""
    for target_node in target_nodes:
        try:
            distance = nx.shortest_path_length(graph, source_node, target_node)
        except nx.NetworkXNoPath:
            continue
        if distance <= max_depth and (best_distance is None or distance < best_distance):
            best_distance = int(distance)
            best_target = target_node

    if best_distance is None:
        return 0.0, None, ""
    return 1.0 / best_distance, best_distance, best_target


def build_reference_sets(
    graph: nx.DiGraph,
    branch_records: list[dict],
    trunk_subtopic_records: list[dict],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build branch reference sets and aggregate trunk-subtopic reference sets."""
    branch_references = {
        record["branch_id"]: set(graph.successors(record["node_id"]))
        for record in branch_records
    }
    trunk_references = {}
    for record in trunk_subtopic_records:
        references: set[str] = set()
        for member_node_id in record["member_node_ids"]:
            references.update(graph.successors(member_node_id))
        trunk_references[record["trunk_subtopic_id"]] = references
    return branch_references, trunk_references


def build_assignment_outputs(
    graph: nx.DiGraph,
    branch_records: list[dict],
    trunk_subtopic_records: list[dict],
    w_path: float,
    w_text: float,
    w_references: float,
    max_path_depth: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score each branch against each main trunk subtopic."""
    vector_records = [
        {"doc_id": record["branch_id"], "tokens": record["tokens"]}
        for record in branch_records
    ] + [
        {"doc_id": record["trunk_subtopic_id"], "tokens": record["tokens"]}
        for record in trunk_subtopic_records
    ]
    vectors = build_document_vectors(vector_records, "doc_id")
    branch_references, trunk_references = build_reference_sets(
        graph,
        branch_records,
        trunk_subtopic_records,
    )

    score_rows = []
    for branch_record in branch_records:
        for trunk_record in trunk_subtopic_records:
            path_score, path_distance, closest_trunk_node = citation_path_proximity(
                graph,
                branch_record["node_id"],
                trunk_record["member_node_ids"],
                max_path_depth,
            )
            text_similarity = cosine_similarity(
                vectors[branch_record["branch_id"]],
                vectors[trunk_record["trunk_subtopic_id"]],
            )
            shared_reference_similarity = overlap_cosine(
                branch_references[branch_record["branch_id"]],
                trunk_references[trunk_record["trunk_subtopic_id"]],
            )
            branch_to_trunk_score = (
                (w_path * path_score)
                + (w_text * text_similarity)
                + (w_references * shared_reference_similarity)
            )
            score_rows.append(
                {
                    "branch_id": branch_record["branch_id"],
                    "node_id": branch_record["node_id"],
                    "ToS": branch_record["ToS"],
                    "sap_rank": branch_record["sap_rank"],
                    "title": branch_record["title"],
                    "year": branch_record["year"],
                    "journal": branch_record["journal"],
                    "source_title": branch_record["source_title"],
                    "has_article_metadata": branch_record["has_article_metadata"],
                    "token_count": branch_record["token_count"],
                    "parent_trunk_subtopic_id": trunk_record["trunk_subtopic_id"],
                    "parent_trunk_subtopic_label": trunk_record["trunk_subtopic_label"],
                    "parent_member_count": trunk_record["member_count"],
                    "branch_to_trunk_score": round(branch_to_trunk_score, 6),
                    "citation_path_proximity": round(path_score, 6),
                    "citation_path_distance": path_distance if path_distance is not None else "",
                    "closest_trunk_node": closest_trunk_node,
                    "text_similarity_to_trunk": round(text_similarity, 6),
                    "shared_reference_similarity": round(shared_reference_similarity, 6),
                    "branch_reference_count": len(branch_references[branch_record["branch_id"]]),
                    "trunk_reference_count": len(trunk_references[trunk_record["trunk_subtopic_id"]]),
                }
            )

    scores_df = pd.DataFrame(score_rows)
    assignments_df = (
        scores_df.sort_values(
            [
                "branch_id",
                "branch_to_trunk_score",
                "citation_path_proximity",
                "text_similarity_to_trunk",
            ],
            ascending=[True, False, False, False],
        )
        .groupby("branch_id", as_index=False)
        .first()
        .sort_values(
            ["parent_trunk_subtopic_id", "ToS", "sap_rank"],
            ascending=[True, True, False],
        )
    )
    assignments_df["assignment_confidence"] = assignments_df[
        "branch_to_trunk_score"
    ].apply(classify_assignment_confidence)
    summary_df = (
        assignments_df.groupby(
            ["parent_trunk_subtopic_id", "parent_trunk_subtopic_label"],
            as_index=False,
        )
        .agg(
            branch_count=("branch_id", "count"),
            mean_assignment_score=("branch_to_trunk_score", "mean"),
            mean_path_proximity=("citation_path_proximity", "mean"),
            mean_text_similarity=("text_similarity_to_trunk", "mean"),
            mean_shared_reference_similarity=("shared_reference_similarity", "mean"),
            strong_assignments=(
                "assignment_confidence",
                lambda values: sum(value == "strong" for value in values),
            ),
            moderate_assignments=(
                "assignment_confidence",
                lambda values: sum(value == "moderate" for value in values),
            ),
            weak_assignments=(
                "assignment_confidence",
                lambda values: sum(value == "weak" for value in values),
            ),
            unassigned=(
                "assignment_confidence",
                lambda values: sum(value == "unassigned" for value in values),
            ),
            member_node_ids=("node_id", lambda values: "; ".join(map(str, values))),
        )
        .sort_values("parent_trunk_subtopic_id")
    )
    return assignments_df, scores_df, summary_df


def classify_assignment_confidence(score: float) -> str:
    """Classify assignment strength for interpretation instead of forcing meaning."""
    score = float(score)
    if score >= 0.25:
        return "strong"
    if score >= 0.08:
        return "moderate"
    if score > 0:
        return "weak"
    return "unassigned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign SAP branch papers to the main trunk subtopics."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--trunk-subtopics", type=Path, default=DEFAULT_TRUNK_SUBTOPICS)
    parser.add_argument(
        "--branch-members",
        type=Path,
        default=None,
        help=(
            "Optional branch-member CSV from an experimental detector. "
            "When omitted, the script uses SAP baseline branch labels from the graph."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--w-path", type=float, default=0.40)
    parser.add_argument("--w-text", type=float, default=0.35)
    parser.add_argument("--w-references", type=float, default=0.25)
    parser.add_argument("--max-path-depth", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_path, w_text, w_references = normalize_weights(
        args.w_path,
        args.w_text,
        args.w_references,
    )
    graph = build_sap_graph(args.citations, args.articles)
    if args.branch_members is None:
        branch_records = extract_branch_records(graph)
    else:
        branch_records = extract_branch_records_from_members(graph, args.branch_members)
    trunk_subtopics_df = load_main_trunk_subtopics(args.trunk_subtopics)
    trunk_subtopic_records = build_trunk_subtopic_records(graph, trunk_subtopics_df)
    assignments_df, scores_df, summary_df = build_assignment_outputs(
        graph,
        branch_records,
        trunk_subtopic_records,
        w_path,
        w_text,
        w_references,
        args.max_path_depth,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(args.output_dir / "branch_trunk_assignments.csv", index=False)
    scores_df.to_csv(args.output_dir / "branch_trunk_assignment_scores.csv", index=False)
    summary_df.to_csv(args.output_dir / "branch_trunk_assignment_summary.csv", index=False)

    print(f"Branch papers: {len(branch_records):,}")
    print(f"Parent trunk subtopics: {len(trunk_subtopic_records):,}")
    print(
        "Weights used: "
        f"path={w_path:.2f}, text={w_text:.2f}, references={w_references:.2f}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
