from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import math
from pathlib import Path

import pandas as pd

from branch_trend_scores import (
    DEFAULT_ARTICLES,
    DEFAULT_BRANCH_ASSIGNMENTS,
    DEFAULT_BRANCH_MEMBERS,
    DEFAULT_CITATIONS,
    branch_id_from_member,
    normalize_weights,
)
from root_tfidf_similarity import build_sap_graph, cosine_similarity, tokenize
from trunk_visualization_metrics import normalize_series


DEFAULT_OUTPUT_DIR = Path("outputs/branch_member_roles")
DEFAULT_REFERENCE_NOVELTY_MEMBERS = Path(
    "outputs/branch_reference_novelty/branch_reference_novelty_members.csv"
)

METHOD_BACKGROUND_TERMS = {
    "bibliometric",
    "checklist",
    "equation",
    "method",
    "methodological",
    "methodology",
    "modeling",
    "modelling",
    "qualitative",
    "reporting",
    "research method",
    "structural equation",
}
NON_SCHOLARLY_SOURCE_TERMS = {
    "guardian",
    "newspaper",
}
DIRECT_TOPIC_TERMS = {
    "art as a product",
    "artist as marketer",
    "entrepreneurial market creation",
    "entrepreneurial marketing",
    "marketing and entrepreneurship",
    "marketing/entrepreneurship",
    "market entrepreneurship",
    "nonprofit arts",
}


def clean_text(value) -> str:
    """Normalize missing CSV/graph text values to an empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    if text.strip().lower() == "nan":
        return ""
    return text


def build_branch_member_table(graph, branch_members_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Leiden/cohesive members with graph metadata and branch ids."""
    rows = []
    for row in branch_members_df.sort_values(["branch_label", "branch_rank"]).itertuples(
        index=False
    ):
        node_id = str(row.node_id)
        attrs = graph.nodes[node_id]
        title = clean_text(attrs.get("title")) or clean_text(getattr(row, "title", ""))
        abstract = clean_text(attrs.get("abstract"))
        text = f"{title} {abstract}".strip()
        tokens = tokenize(text)
        year = pd.to_numeric(attrs.get("year"), errors="coerce")
        if pd.isna(year):
            year = pd.to_numeric(getattr(row, "year", ""), errors="coerce")
        rows.append(
            {
                "branch_id": branch_id_from_member(row),
                "branch_label": str(row.branch_label),
                "branch_rank": int(row.branch_rank),
                "node_id": node_id,
                "year": int(float(year)) if pd.notna(year) else "",
                "title": title,
                "abstract": abstract,
                "journal": clean_text(attrs.get("journal"))
                or clean_text(getattr(row, "journal", "")),
                "source_title": clean_text(attrs.get("source_title"))
                or clean_text(getattr(row, "source_title", "")),
                "doi": clean_text(attrs.get("doi")) or clean_text(getattr(row, "doi", "")),
                "has_article_metadata": bool(
                    attrs.get("has_article_metadata", getattr(row, "has_article_metadata", False))
                ),
                "token_count": len(tokens),
                "tokens": tokens,
            }
        )
    return pd.DataFrame(rows)


def compute_tfidf_vectors(records: list[dict]) -> dict[str, dict[str, float]]:
    """Compute lightweight TF-IDF vectors for branch-member texts."""
    document_tokens = {record["branch_id"]: record["tokens"] for record in records}
    document_count = len(document_tokens)
    document_frequency = Counter()
    for tokens in document_tokens.values():
        document_frequency.update(set(tokens))

    vectors = {}
    for branch_id, tokens in document_tokens.items():
        term_counts = Counter(tokens)
        total_terms = sum(term_counts.values())
        vector = {}
        if total_terms:
            for term, count in term_counts.items():
                term_frequency = count / total_terms
                inverse_document_frequency = math.log(
                    (1 + document_count) / (1 + document_frequency[term])
                ) + 1
                vector[term] = term_frequency * inverse_document_frequency
        vectors[branch_id] = vector
    return vectors


def centroid_vector(vectors: list[dict[str, float]]) -> dict[str, float]:
    """Average sparse vectors into a branch centroid."""
    if not vectors:
        return {}
    centroid = Counter()
    for vector in vectors:
        centroid.update(vector)
    return {term: value / len(vectors) for term, value in centroid.items()}


def add_semantic_metrics(members_df: pd.DataFrame) -> pd.DataFrame:
    """Add similarity to branch centroid and branch-level top terms."""
    records = members_df.to_dict("records")
    vectors = compute_tfidf_vectors(records)
    rows = []
    for branch_label, group in members_df.groupby("branch_label"):
        branch_ids = group["branch_id"].tolist()
        usable_vectors = [
            vectors[branch_id]
            for branch_id in branch_ids
            if vectors[branch_id]
        ]
        centroid = centroid_vector(usable_vectors)
        top_terms = "; ".join(
            term for term, _ in sorted(centroid.items(), key=lambda item: item[1], reverse=True)[:12]
        )
        for row in group.itertuples(index=False):
            vector = vectors[row.branch_id]
            rows.append(
                {
                    "branch_id": row.branch_id,
                    "semantic_similarity_to_branch": cosine_similarity(vector, centroid),
                    "branch_top_terms": top_terms,
                }
            )
    semantic_df = pd.DataFrame(rows)
    return members_df.merge(semantic_df, on="branch_id", how="left")


def add_assignment_metrics(
    members_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach branch-to-trunk assignment signals."""
    columns = [
        "branch_id",
        "parent_trunk_subtopic_id",
        "parent_trunk_subtopic_label",
        "branch_to_trunk_score",
        "citation_path_proximity",
        "text_similarity_to_trunk",
        "shared_reference_similarity",
        "assignment_confidence",
    ]
    return members_df.merge(assignments_df[columns], on="branch_id", how="left")


def add_reference_novelty(
    members_df: pd.DataFrame,
    reference_novelty_members: Path | None,
) -> pd.DataFrame:
    """Attach optional reference novelty diagnostics."""
    if reference_novelty_members is None or not reference_novelty_members.exists():
        members_df["reference_novelty"] = ""
        members_df["novelty_supported"] = False
        return members_df
    novelty_df = pd.read_csv(reference_novelty_members)
    columns = [
        "branch_id",
        "reference_count",
        "same_domain_candidate_count",
        "reference_novelty",
        "novelty_supported",
    ]
    return members_df.merge(novelty_df[columns], on="branch_id", how="left")


def has_any_term(text: str, terms: set[str]) -> bool:
    """Check whether any phrase appears in text."""
    text = text.lower()
    return any(term in text for term in terms)


def add_quality_flags(
    members_df: pd.DataFrame,
    current_year: int,
    recent_window_years: int,
    min_tokens: int,
) -> pd.DataFrame:
    """Add metadata, recency, and background flags."""
    df = members_df.copy()
    df["year_numeric"] = pd.to_numeric(df["year"], errors="coerce")
    df["paper_age"] = current_year - df["year_numeric"]
    df["is_recent"] = df["paper_age"].between(0, recent_window_years - 1, inclusive="both")
    df["missing_title"] = df["title"].fillna("").str.strip().eq("")
    df["missing_abstract"] = df["abstract"].fillna("").str.strip().eq("")
    df["low_text_information"] = df["token_count"].fillna(0).astype(int) < min_tokens
    df["metadata_quality"] = "complete"
    df.loc[df["missing_abstract"], "metadata_quality"] = "title_only"
    df.loc[df["missing_title"] | df["low_text_information"], "metadata_quality"] = "weak"
    searchable_text = (
        df["title"].fillna("")
        + " "
        + df["journal"].fillna("")
        + " "
        + df["source_title"].fillna("")
    )
    df["is_methodological_or_background"] = searchable_text.apply(
        lambda text: has_any_term(text, METHOD_BACKGROUND_TERMS)
    )
    df["has_bibliographic_warning"] = searchable_text.apply(
        lambda text: has_any_term(text, NON_SCHOLARLY_SOURCE_TERMS)
    )
    direct_topic_text = (
        df["title"].fillna("")
        + " "
        + df["abstract"].fillna("")
        + " "
        + df["journal"].fillna("")
        + " "
        + df["source_title"].fillna("")
    )
    df["has_direct_topic_signal"] = direct_topic_text.apply(
        lambda text: has_any_term(text, DIRECT_TOPIC_TERMS)
    )
    return df


def score_member_roles(
    members_df: pd.DataFrame,
    semantic_weight: float,
    recency_weight: float,
    trunk_weight: float,
) -> pd.DataFrame:
    """Score branch members as potential core papers."""
    df = members_df.copy()
    df["semantic_local_score"] = (
        df.groupby("branch_label", group_keys=False)["semantic_similarity_to_branch"]
        .apply(normalize_series)
        .astype(float)
    )
    df["recency_score"] = df["is_recent"].astype(float)
    df["trunk_connection_score"] = (
        df.groupby("branch_label", group_keys=False)["branch_to_trunk_score"]
        .apply(lambda values: normalize_series(pd.to_numeric(values, errors="coerce").fillna(0)))
        .astype(float)
    )
    df["branch_core_score"] = (
        (semantic_weight * df["semantic_local_score"])
        + (recency_weight * df["recency_score"])
        + (trunk_weight * df["trunk_connection_score"])
    )
    return df


def classify_member_role(row, core_score_threshold: float, min_semantic_similarity: float) -> tuple[str, str]:
    """Classify a branch member into a role and a concise reason."""
    if row.missing_title or row.low_text_information:
        return "missing_metadata", "title/text information is insufficient for interpretation"
    if row.has_bibliographic_warning:
        return "background_methodological", "bibliographic warning or non-scholarly source"
    if row.is_methodological_or_background:
        return "background_methodological", "methodological or background-oriented title/source"
    if (
        row.has_direct_topic_signal
        and row.branch_core_score >= core_score_threshold
        and row.semantic_similarity_to_branch >= min_semantic_similarity
    ):
        return "core", "direct topical signal and strong branch fit"
    if (
        row.has_direct_topic_signal
        and row.semantic_local_score >= 0.50
        and row.semantic_similarity_to_branch >= min_semantic_similarity
    ):
        return "core", "direct topical signal and strong semantic fit"
    if (
        row.has_direct_topic_signal
        and row.semantic_similarity_to_branch >= 0.20
        and row.branch_core_score >= 0.20
    ):
        return "core", "direct topical signal and adequate thematic fit"
    if (
        row.assignment_confidence in {"strong", "moderate"}
        and row.semantic_similarity_to_branch >= min_semantic_similarity
        and row.has_direct_topic_signal
    ):
        return "core", "direct topical signal with strong/moderate trunk connection"
    return "peripheral", "related to branch but weaker as a defining paper"


def assign_member_roles(
    members_df: pd.DataFrame,
    core_score_threshold: float,
    min_semantic_similarity: float,
) -> pd.DataFrame:
    """Assign final branch roles."""
    df = members_df.copy()
    roles = df.apply(
        lambda row: classify_member_role(
            row,
            core_score_threshold,
            min_semantic_similarity,
        ),
        axis=1,
    )
    df["branch_member_role"] = [role for role, _ in roles]
    df["role_reason"] = [reason for _, reason in roles]
    return df


def summarize_roles(roles_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize role distribution and core-paper labels by branch."""
    rows = []
    for branch_label, group in roles_df.groupby("branch_label"):
        role_counts = group["branch_member_role"].value_counts().to_dict()
        core_group = group[group["branch_member_role"] == "core"]
        rows.append(
            {
                "branch_label": branch_label,
                "paper_count": len(group),
                "core_count": role_counts.get("core", 0),
                "peripheral_count": role_counts.get("peripheral", 0),
                "background_methodological_count": role_counts.get(
                    "background_methodological",
                    0,
                ),
                "missing_metadata_count": role_counts.get("missing_metadata", 0),
                "core_share": role_counts.get("core", 0) / len(group),
                "mean_core_score": group["branch_core_score"].mean(),
                "mean_semantic_similarity": group["semantic_similarity_to_branch"].mean(),
                "branch_top_terms": str(group["branch_top_terms"].iloc[0]),
                "core_node_ids": "; ".join(core_group["node_id"].astype(str).tolist()),
                "core_titles": "; ".join(core_group["title"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["core_share", "core_count"], ascending=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify branch papers as core, peripheral, background, or missing metadata."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--branch-members", type=Path, default=DEFAULT_BRANCH_MEMBERS)
    parser.add_argument(
        "--branch-assignments",
        type=Path,
        default=DEFAULT_BRANCH_ASSIGNMENTS,
    )
    parser.add_argument(
        "--reference-novelty-members",
        type=Path,
        default=DEFAULT_REFERENCE_NOVELTY_MEMBERS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    parser.add_argument("--recent-window-years", type=int, default=5)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--semantic-weight", type=float, default=0.50)
    parser.add_argument("--recency-weight", type=float, default=0.30)
    parser.add_argument("--trunk-weight", type=float, default=0.20)
    parser.add_argument("--core-score-threshold", type=float, default=0.55)
    parser.add_argument("--min-semantic-similarity", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic_weight, recency_weight, trunk_weight = normalize_weights(
        args.semantic_weight,
        args.recency_weight,
        args.trunk_weight,
    )
    graph = build_sap_graph(args.citations, args.articles)
    branch_members_df = pd.read_csv(args.branch_members)
    assignments_df = pd.read_csv(args.branch_assignments)
    members_df = build_branch_member_table(graph, branch_members_df)
    members_df = add_semantic_metrics(members_df)
    members_df = add_assignment_metrics(members_df, assignments_df)
    members_df = add_reference_novelty(members_df, args.reference_novelty_members)
    members_df = add_quality_flags(
        members_df,
        args.current_year,
        args.recent_window_years,
        args.min_tokens,
    )
    members_df = score_member_roles(
        members_df,
        semantic_weight,
        recency_weight,
        trunk_weight,
    )
    roles_df = assign_member_roles(
        members_df,
        args.core_score_threshold,
        args.min_semantic_similarity,
    )
    summary_df = summarize_roles(roles_df)

    output_columns = [
        "branch_label",
        "branch_rank",
        "branch_member_role",
        "role_reason",
        "branch_core_score",
        "semantic_similarity_to_branch",
        "semantic_local_score",
        "recency_score",
        "trunk_connection_score",
        "assignment_confidence",
        "branch_to_trunk_score",
        "parent_trunk_subtopic_id",
        "metadata_quality",
        "has_direct_topic_signal",
        "is_methodological_or_background",
        "has_bibliographic_warning",
        "reference_novelty",
        "novelty_supported",
        "node_id",
        "year",
        "title",
        "journal",
        "source_title",
        "branch_top_terms",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roles_df[output_columns].to_csv(
        args.output_dir / "branch_member_roles.csv",
        index=False,
    )
    summary_df.to_csv(args.output_dir / "branch_member_role_summary.csv", index=False)

    print(f"Branch papers: {len(roles_df):,}")
    print(f"Branches: {len(summary_df):,}")
    print(
        "Weights used: "
        f"semantic={semantic_weight:.2f}, recency={recency_weight:.2f}, "
        f"trunk={trunk_weight:.2f}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
