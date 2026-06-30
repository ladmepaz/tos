from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import math
from pathlib import Path

import pandas as pd

from branch_leiden_comparison import mean_pairwise_text_similarity
from root_tfidf_similarity import build_sap_graph, tokenize
from trunk_visualization_metrics import normalize_series


DEFAULT_CITATIONS = Path("data/processed/bibfusion/All_Citation_tidy.csv")
DEFAULT_ARTICLES = Path("data/processed/bibfusion/All_Articles_tidy.csv")
DEFAULT_BRANCH_MEMBERS = Path(
    "outputs/branch_leiden_comparison/leiden_cohesive_branch_members.csv"
)
DEFAULT_BRANCH_ASSIGNMENTS = Path(
    "outputs/experiments/branch_assignment_leiden_cohesive/branch_trunk_assignments.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/branch_trends")


def branch_id_from_member(row) -> str:
    """Build the same branch id used by branch_trunk_assignment.py."""
    return f"{row.branch_label}_{int(row.branch_rank):02d}"


def compute_member_metrics(
    graph,
    branch_members_df: pd.DataFrame,
    current_year: int,
    recent_window_years: int,
) -> pd.DataFrame:
    """Add paper-level trend inputs for the selected branch papers."""
    rows = []
    for row in branch_members_df.sort_values(["branch_label", "branch_rank"]).itertuples(
        index=False
    ):
        node_id = str(row.node_id)
        attrs = graph.nodes[node_id]
        publication_year = pd.to_numeric(attrs.get("year"), errors="coerce")
        if pd.isna(publication_year):
            publication_year = pd.to_numeric(row.year, errors="coerce")
        if pd.isna(publication_year):
            publication_year = current_year

        publication_year = int(float(publication_year))
        paper_age = max(current_year - publication_year, 0)
        years_since_publication = max(current_year - publication_year + 1, 1)
        citations_received = float(graph.in_degree(node_id))
        citation_velocity = citations_received / years_since_publication
        recent_activity_score = max(
            0.0,
            1.0 - (paper_age / max(recent_window_years, 1)),
        )

        rows.append(
            {
                "branch_id": branch_id_from_member(row),
                "branch_label": row.branch_label,
                "branch_rank": int(row.branch_rank),
                "node_id": node_id,
                "year": publication_year,
                "paper_age": paper_age,
                "citations_received": citations_received,
                "citation_velocity": citation_velocity,
                "recent_activity_score": recent_activity_score,
                "title": attrs.get("title") or getattr(row, "title", ""),
                "journal": attrs.get("journal") or getattr(row, "journal", ""),
                "source_title": attrs.get("source_title") or getattr(row, "source_title", ""),
                "has_article_metadata": attrs.get(
                    "has_article_metadata",
                    getattr(row, "has_article_metadata", False),
                ),
            }
        )

    members_df = pd.DataFrame(rows)
    members_df["log_citation_velocity"] = members_df["citation_velocity"].apply(math.log1p)
    members_df["citation_velocity_score"] = normalize_series(
        members_df["log_citation_velocity"]
    )
    return members_df


def aggregate_top_terms(graph, node_ids: list[str], top_n: int) -> str:
    """Summarize a branch using title/abstract terms."""
    terms = Counter()
    for node_id in node_ids:
        attrs = graph.nodes[node_id]
        terms.update(tokenize(f"{attrs.get('title') or ''} {attrs.get('abstract') or ''}"))
    return "; ".join(term for term, _ in terms.most_common(top_n))


def build_branch_trend_scores(
    graph,
    member_metrics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    w_structural: float,
    w_semantic: float,
    w_recent: float,
    w_velocity: float,
    top_terms: int,
) -> pd.DataFrame:
    """Compute one trend score per branch."""
    assignment_lookup = assignments_df.set_index("branch_id")
    rows = []
    for branch_label, group in member_metrics_df.groupby("branch_label"):
        branch_ids = group["branch_id"].tolist()
        assignment_group = assignment_lookup.loc[
            assignment_lookup.index.intersection(branch_ids)
        ]
        node_ids = group["node_id"].astype(str).tolist()

        structural_connection_to_trunk = (
            float(assignment_group["branch_to_trunk_score"].mean())
            if not assignment_group.empty
            else 0.0
        )
        mean_path_proximity = (
            float(assignment_group["citation_path_proximity"].mean())
            if not assignment_group.empty
            else 0.0
        )
        strong_assignments = (
            int((assignment_group["assignment_confidence"] == "strong").sum())
            if not assignment_group.empty
            else 0
        )
        moderate_assignments = (
            int((assignment_group["assignment_confidence"] == "moderate").sum())
            if not assignment_group.empty
            else 0
        )
        dominant_parent = (
            assignment_group["parent_trunk_subtopic_id"].mode().iloc[0]
            if not assignment_group.empty
            else ""
        )
        parent_counts = (
            assignment_group["parent_trunk_subtopic_id"].value_counts().sort_index()
            if not assignment_group.empty
            else pd.Series(dtype=int)
        )
        parent_score_sums = (
            assignment_group.groupby("parent_trunk_subtopic_id")[
                "branch_to_trunk_score"
            ]
            .sum()
            .sort_values(ascending=False)
            if not assignment_group.empty
            else pd.Series(dtype=float)
        )
        score_weighted_parent = (
            str(parent_score_sums.index[0]) if not parent_score_sums.empty else ""
        )

        rows.append(
            {
                "branch_label": branch_label,
                "paper_count": len(group),
                "dominant_parent_trunk_subtopic_id": dominant_parent,
                "score_weighted_parent_trunk_subtopic_id": score_weighted_parent,
                "parent_trunk_distribution": "; ".join(
                    f"{parent}:{count}" for parent, count in parent_counts.items()
                ),
                "parent_trunk_score_distribution": "; ".join(
                    f"{parent}:{score:.6f}" for parent, score in parent_score_sums.items()
                ),
                "structural_connection_to_trunk": structural_connection_to_trunk,
                "mean_path_proximity": mean_path_proximity,
                "strong_assignments": strong_assignments,
                "moderate_assignments": moderate_assignments,
                "semantic_coherence": mean_pairwise_text_similarity(graph, node_ids),
                "mean_publication_year": float(group["year"].mean()),
                "newest_publication_year": int(group["year"].max()),
                "oldest_publication_year": int(group["year"].min()),
                "recent_activity": float(group["recent_activity_score"].mean()),
                "recent_paper_share": float((group["recent_activity_score"] > 0).mean()),
                "citations_received": float(group["citations_received"].sum()),
                "mean_citation_velocity": float(group["citation_velocity"].mean()),
                "max_citation_velocity": float(group["citation_velocity"].max()),
                "top_terms": aggregate_top_terms(graph, node_ids, top_terms),
                "member_node_ids": "; ".join(node_ids),
            }
        )

    branch_df = pd.DataFrame(rows)
    branch_df["structural_score"] = normalize_series(
        branch_df["structural_connection_to_trunk"]
    )
    branch_df["semantic_score"] = normalize_series(branch_df["semantic_coherence"])
    branch_df["recent_score"] = normalize_series(branch_df["recent_activity"])
    branch_df["velocity_score"] = normalize_series(
        branch_df["mean_citation_velocity"].apply(math.log1p)
    )
    branch_df["branch_trend_score"] = (
        (w_structural * branch_df["structural_score"])
        + (w_semantic * branch_df["semantic_score"])
        + (w_recent * branch_df["recent_score"])
        + (w_velocity * branch_df["velocity_score"])
    )
    branch_df["branch_trend_rank"] = branch_df["branch_trend_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return branch_df.sort_values("branch_trend_rank")


def normalize_weights(*weights: float) -> list[float]:
    """Normalize positive weights so they sum to one."""
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("At least one trend-score weight must be positive.")
    return [weight / weight_sum for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score Leiden/cohesive branches as trend candidates."
    )
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--branch-members", type=Path, default=DEFAULT_BRANCH_MEMBERS)
    parser.add_argument(
        "--branch-assignments",
        type=Path,
        default=DEFAULT_BRANCH_ASSIGNMENTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--current-year", type=int, default=date.today().year)
    parser.add_argument("--recent-window-years", type=int, default=7)
    parser.add_argument("--w-structural", type=float, default=0.30)
    parser.add_argument("--w-semantic", type=float, default=0.25)
    parser.add_argument("--w-recent", type=float, default=0.25)
    parser.add_argument("--w-velocity", type=float, default=0.20)
    parser.add_argument("--top-terms", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_structural, w_semantic, w_recent, w_velocity = normalize_weights(
        args.w_structural,
        args.w_semantic,
        args.w_recent,
        args.w_velocity,
    )
    graph = build_sap_graph(args.citations, args.articles)
    branch_members_df = pd.read_csv(args.branch_members)
    assignments_df = pd.read_csv(args.branch_assignments)
    member_metrics_df = compute_member_metrics(
        graph,
        branch_members_df,
        args.current_year,
        args.recent_window_years,
    )
    trend_scores_df = build_branch_trend_scores(
        graph,
        member_metrics_df,
        assignments_df,
        w_structural,
        w_semantic,
        w_recent,
        w_velocity,
        args.top_terms,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_metrics_df.to_csv(args.output_dir / "branch_trend_members.csv", index=False)
    trend_scores_df.to_csv(args.output_dir / "branch_trend_scores.csv", index=False)

    print(f"Branch papers: {len(member_metrics_df):,}")
    print(f"Branches: {len(trend_scores_df):,}")
    print(
        "Weights used: "
        f"structural={w_structural:.2f}, semantic={w_semantic:.2f}, "
        f"recent={w_recent:.2f}, velocity={w_velocity:.2f}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
