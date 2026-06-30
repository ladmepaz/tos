from __future__ import annotations

import argparse
from datetime import date
import math
from pathlib import Path
import re

import pandas as pd

from branch_trend_scores import (
    DEFAULT_ARTICLES,
    DEFAULT_BRANCH_ASSIGNMENTS,
    DEFAULT_BRANCH_MEMBERS,
    DEFAULT_CITATIONS,
    branch_id_from_member,
)
from root_tfidf_similarity import build_sap_graph
from trunk_visualization_metrics import normalize_series


DEFAULT_OUTPUT_DIR = Path("outputs/experiments/branch_reference_novelty")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def parse_year_from_node_id(node_id: str) -> int | None:
    """Extract a year from an SR-like node id when metadata is missing."""
    match = YEAR_PATTERN.search(str(node_id))
    if not match:
        return None
    return int(match.group(0))


def node_year(graph, node_id: str) -> int | None:
    """Return the best available publication year for a graph node."""
    value = graph.nodes[node_id].get("year")
    year = pd.to_numeric(value, errors="coerce")
    if pd.notna(year):
        return int(float(year))
    return parse_year_from_node_id(node_id)


def filtered_references(
    graph,
    node_id: str,
    focal_year: int | None,
    reference_window_years: int | None,
    include_missing_reference_years: bool,
) -> set[str]:
    """Return cited references, optionally limited to a recent reference window."""
    references = set()
    for reference in graph.successors(node_id):
        reference_year = node_year(graph, reference)
        if focal_year is None or reference_window_years is None:
            references.add(reference)
            continue
        if reference_year is None:
            if include_missing_reference_years:
                references.add(reference)
            continue
        if focal_year - reference_window_years <= reference_year <= focal_year:
            references.add(reference)
    return references


def jaccard_overlap(source_refs: set[str], target_refs: set[str]) -> float:
    """Compute the reference-overlap score used by the novelty indicator."""
    union = source_refs | target_refs
    if not union:
        return 0.0
    return len(source_refs & target_refs) / len(union)


def build_branch_member_table(branch_members_df: pd.DataFrame) -> pd.DataFrame:
    """Add branch_id to the Leiden/cohesive member table."""
    members_df = branch_members_df.copy()
    members_df["branch_id"] = [
        branch_id_from_member(row)
        for row in members_df.itertuples(index=False)
    ]
    return members_df


def same_domain_candidates(
    graph,
    focal_node: str,
    focal_year: int | None,
    focal_references: set[str],
    reference_cache: dict[str, set[str]],
    min_shared_references: int,
    co_citing_window_years: int | None,
) -> list[str]:
    """Approximate same-domain prior papers through shared cited references."""
    candidates = []
    if not focal_references:
        return candidates

    for candidate_node, candidate_refs in reference_cache.items():
        if candidate_node == focal_node:
            continue
        candidate_year = node_year(graph, candidate_node)
        if focal_year is not None:
            if candidate_year is None or candidate_year >= focal_year:
                continue
            if (
                co_citing_window_years is not None
                and candidate_year < focal_year - co_citing_window_years
            ):
                continue

        shared_count = len(focal_references & candidate_refs)
        if shared_count >= min_shared_references:
            candidates.append(candidate_node)

    return candidates


def compute_reference_cache(
    graph,
    reference_window_years: int | None,
    include_missing_reference_years: bool,
) -> dict[str, set[str]]:
    """Precompute reference sets for all graph nodes."""
    cache = {}
    for node_id in graph.nodes:
        focal_year = node_year(graph, node_id)
        cache[node_id] = filtered_references(
            graph,
            node_id,
            focal_year,
            reference_window_years,
            include_missing_reference_years,
        )
    return cache


def compute_member_novelty(
    graph,
    members_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    reference_window_years: int | None,
    co_citing_window_years: int | None,
    min_shared_references: int,
    min_same_domain_candidates: int,
    include_missing_reference_years: bool,
) -> pd.DataFrame:
    """Compute Matsumoto-style reference novelty for branch papers."""
    reference_cache = compute_reference_cache(
        graph,
        reference_window_years,
        include_missing_reference_years,
    )
    assignment_lookup = assignments_df.set_index("branch_id")
    rows = []

    for row in members_df.sort_values(["branch_label", "branch_rank"]).itertuples(
        index=False
    ):
        node_id = str(row.node_id)
        focal_year = node_year(graph, node_id)
        focal_references = reference_cache[node_id]
        candidates = same_domain_candidates(
            graph,
            node_id,
            focal_year,
            focal_references,
            reference_cache,
            min_shared_references,
            co_citing_window_years,
        )
        overlap_scores = [
            jaccard_overlap(focal_references, reference_cache[candidate])
            for candidate in candidates
        ]
        mean_overlap = (
            sum(overlap_scores) / len(overlap_scores)
            if overlap_scores
            else math.nan
        )
        novelty = 1 - mean_overlap if not math.isnan(mean_overlap) else math.nan
        supported = len(candidates) >= min_same_domain_candidates
        assignment = (
            assignment_lookup.loc[row.branch_id]
            if row.branch_id in assignment_lookup.index
            else None
        )

        rows.append(
            {
                "branch_id": row.branch_id,
                "branch_label": row.branch_label,
                "branch_rank": int(row.branch_rank),
                "node_id": node_id,
                "year": focal_year if focal_year is not None else "",
                "title": getattr(row, "title", ""),
                "reference_count": len(focal_references),
                "same_domain_candidate_count": len(candidates),
                "mean_reference_overlap": round(mean_overlap, 6)
                if not math.isnan(mean_overlap)
                else "",
                "reference_novelty": round(novelty, 6)
                if not math.isnan(novelty)
                else "",
                "novelty_supported": supported,
                "parent_trunk_subtopic_id": assignment["parent_trunk_subtopic_id"]
                if assignment is not None
                else "",
                "branch_to_trunk_score": assignment["branch_to_trunk_score"]
                if assignment is not None
                else "",
                "assignment_confidence": assignment["assignment_confidence"]
                if assignment is not None
                else "",
                "same_domain_candidates": "; ".join(candidates[:50]),
            }
        )

    novelty_df = pd.DataFrame(rows)
    numeric_novelty = pd.to_numeric(
        novelty_df["reference_novelty"],
        errors="coerce",
    )
    novelty_df["reference_novelty_score"] = normalize_series(
        numeric_novelty.fillna(numeric_novelty.min() if numeric_novelty.notna().any() else 0)
    )
    return novelty_df


def summarize_branch_novelty(
    member_novelty_df: pd.DataFrame,
    current_year: int,
    recent_window_years: int,
) -> pd.DataFrame:
    """Aggregate paper-level reference novelty into branch-level indicators."""
    rows = []
    numeric_df = member_novelty_df.copy()
    numeric_df["reference_novelty"] = pd.to_numeric(
        numeric_df["reference_novelty"],
        errors="coerce",
    )
    numeric_df["year"] = pd.to_numeric(numeric_df["year"], errors="coerce")
    recent_start_year = current_year - recent_window_years + 1

    for branch_label, group in numeric_df.groupby("branch_label"):
        supported = group[group["novelty_supported"] & group["reference_novelty"].notna()]
        recent_supported = supported[supported["year"] >= recent_start_year]
        rows.append(
            {
                "branch_label": branch_label,
                "paper_count": len(group),
                "supported_paper_count": len(supported),
                "unsupported_paper_count": len(group) - len(supported),
                "mean_reference_novelty": float(supported["reference_novelty"].mean())
                if not supported.empty
                else 0.0,
                "median_reference_novelty": float(supported["reference_novelty"].median())
                if not supported.empty
                else 0.0,
                "max_reference_novelty": float(supported["reference_novelty"].max())
                if not supported.empty
                else 0.0,
                "recent_mean_reference_novelty": float(
                    recent_supported["reference_novelty"].mean()
                )
                if not recent_supported.empty
                else 0.0,
                "high_novelty_share": float((supported["reference_novelty"] >= 0.80).mean())
                if not supported.empty
                else 0.0,
                "mean_same_domain_candidate_count": float(
                    supported["same_domain_candidate_count"].mean()
                )
                if not supported.empty
                else 0.0,
                "member_node_ids": "; ".join(group["node_id"].astype(str).tolist()),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df["branch_reference_novelty_score"] = normalize_series(
        summary_df["recent_mean_reference_novelty"]
    )
    summary_df["branch_reference_novelty_rank"] = summary_df[
        "branch_reference_novelty_score"
    ].rank(method="first", ascending=False).astype(int)
    return summary_df.sort_values("branch_reference_novelty_rank")


def parse_optional_window(value: int) -> int | None:
    """Treat non-positive CLI windows as no filtering."""
    return value if value > 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reference-recombination novelty for ToS branch papers "
            "using a Matsumoto-style overlap indicator."
        )
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
    parser.add_argument("--recent-window-years", type=int, default=5)
    parser.add_argument("--reference-window-years", type=int, default=10)
    parser.add_argument("--co-citing-window-years", type=int, default=10)
    parser.add_argument("--min-shared-references", type=int, default=1)
    parser.add_argument("--min-same-domain-candidates", type=int, default=3)
    parser.add_argument(
        "--exclude-missing-reference-years",
        action="store_true",
        help="Drop cited references whose publication year cannot be inferred.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_sap_graph(args.citations, args.articles)
    members_df = build_branch_member_table(pd.read_csv(args.branch_members))
    assignments_df = pd.read_csv(args.branch_assignments)
    member_novelty_df = compute_member_novelty(
        graph,
        members_df,
        assignments_df,
        parse_optional_window(args.reference_window_years),
        parse_optional_window(args.co_citing_window_years),
        args.min_shared_references,
        args.min_same_domain_candidates,
        not args.exclude_missing_reference_years,
    )
    branch_novelty_df = summarize_branch_novelty(
        member_novelty_df,
        args.current_year,
        args.recent_window_years,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_novelty_df.to_csv(
        args.output_dir / "branch_reference_novelty_members.csv",
        index=False,
    )
    branch_novelty_df.to_csv(
        args.output_dir / "branch_reference_novelty_scores.csv",
        index=False,
    )

    print(f"Branch papers: {len(member_novelty_df):,}")
    print(f"Branches: {len(branch_novelty_df):,}")
    print(
        "Supported novelty scores: "
        f"{int(member_novelty_df['novelty_supported'].sum()):,}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
