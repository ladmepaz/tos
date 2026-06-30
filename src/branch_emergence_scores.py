from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import math
from pathlib import Path

import networkx as nx
import pandas as pd

from branch_leiden_comparison import mean_pairwise_text_similarity
from branch_trend_scores import (
    DEFAULT_ARTICLES,
    DEFAULT_BRANCH_ASSIGNMENTS,
    DEFAULT_BRANCH_MEMBERS,
    DEFAULT_CITATIONS,
    aggregate_top_terms,
    compute_member_metrics,
    normalize_weights,
)
from root_tfidf_similarity import build_sap_graph
from trunk_visualization_metrics import normalize_series


DEFAULT_OUTPUT_DIR = Path("outputs/experiments/branch_emergence")
DEFAULT_REFERENCE_NOVELTY = Path(
    "outputs/experiments/branch_reference_novelty/branch_reference_novelty_scores.csv"
)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe ratio for small branch-level counts."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_parent_distributions(assignments_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize parent-trunk clarity for each branch."""
    rows = []
    for branch_label, group in assignments_df.groupby("ToS"):
        score_by_parent = (
            group.groupby("parent_trunk_subtopic_id")["branch_to_trunk_score"]
            .sum()
            .sort_values(ascending=False)
        )
        count_by_parent = group["parent_trunk_subtopic_id"].value_counts().sort_index()
        total_score = float(score_by_parent.sum())
        parent_score_concentration = (
            float(score_by_parent.iloc[0] / total_score) if total_score > 0 else 0.0
        )
        confidence_share = float(
            group["assignment_confidence"].isin(["strong", "moderate"]).mean()
        )
        rows.append(
            {
                "branch_label": branch_label,
                "dominant_parent_trunk_subtopic_id": count_by_parent.idxmax(),
                "score_weighted_parent_trunk_subtopic_id": score_by_parent.index[0]
                if not score_by_parent.empty
                else "",
                "parent_score_concentration": parent_score_concentration,
                "assignment_confidence_share": confidence_share,
                "parent_trunk_distribution": "; ".join(
                    f"{parent}:{count}" for parent, count in count_by_parent.items()
                ),
                "parent_trunk_score_distribution": "; ".join(
                    f"{parent}:{score:.6f}" for parent, score in score_by_parent.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def publication_agr(year_counts: Counter[int], years: list[int]) -> float:
    """Compute a small-sample average growth rate over yearly counts."""
    if len(years) < 2:
        return 0.0
    growth_rates = []
    for previous_year, current_year in zip(years, years[1:]):
        previous_count = year_counts[previous_year]
        current_count = year_counts[current_year]
        if previous_count == 0:
            growth_rates.append(float(current_count))
        else:
            growth_rates.append((current_count - previous_count) / previous_count)
    return sum(growth_rates) / len(growth_rates) if growth_rates else 0.0


def removal_impact(graph: nx.DiGraph, node_ids: list[str]) -> dict[str, float]:
    """Approximate Xu et al.'s uncertainty-reduction logic with graph disruption."""
    undirected = graph.to_undirected()
    baseline_components = nx.number_connected_components(undirected)
    baseline_giant = max((len(component) for component in nx.connected_components(undirected)), default=0)

    reduced = undirected.copy()
    reduced.remove_nodes_from(node_ids)
    reduced_components = nx.number_connected_components(reduced)
    reduced_giant = max((len(component) for component in nx.connected_components(reduced)), default=0)

    return {
        "component_delta_after_branch_removal": float(
            reduced_components - baseline_components
        ),
        "giant_component_loss_after_branch_removal": float(
            max(0, baseline_giant - reduced_giant - len(node_ids))
        ),
    }


def build_raw_emergence_indicators(
    graph: nx.DiGraph,
    member_metrics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    current_year: int,
    recent_window_years: int,
    top_terms: int,
) -> pd.DataFrame:
    """Compute Xu-inspired raw branch indicators from available ToS data."""
    parent_df = compute_parent_distributions(assignments_df)
    assignment_lookup = assignments_df.set_index("branch_id")
    rows = []

    for branch_label, group in member_metrics_df.groupby("branch_label"):
        node_ids = group["node_id"].astype(str).tolist()
        branch_ids = group["branch_id"].tolist()
        assignment_group = assignment_lookup.loc[
            assignment_lookup.index.intersection(branch_ids)
        ]

        year_counts = Counter(group["year"].astype(int).tolist())
        year_span = list(range(int(group["year"].min()), current_year + 1))
        recent_start_year = current_year - recent_window_years + 1
        previous_start_year = current_year - (2 * recent_window_years) + 1
        recent_count = int((group["year"] >= recent_start_year).sum())
        previous_count = int(
            ((group["year"] >= previous_start_year) & (group["year"] < recent_start_year)).sum()
        )

        selected_subgraph = graph.to_undirected().subgraph(node_ids).copy()
        component_sizes = [
            len(component) for component in nx.connected_components(selected_subgraph)
        ]
        largest_component_fraction = (
            max(component_sizes) / len(node_ids) if node_ids else 0.0
        )
        density = nx.density(selected_subgraph) if len(node_ids) > 1 else 0.0
        impact_after_removal = removal_impact(graph, node_ids)

        rows.append(
            {
                "branch_label": branch_label,
                "paper_count": len(group),
                "recent_window_years": recent_window_years,
                "recent_paper_count": recent_count,
                "previous_window_paper_count": previous_count,
                "recent_paper_share": safe_ratio(recent_count, len(group)),
                "recent_growth_rate": safe_ratio(
                    recent_count - previous_count,
                    max(previous_count, 1),
                ),
                "publication_agr": publication_agr(year_counts, year_span),
                "mean_publication_year": float(group["year"].mean()),
                "newest_publication_year": int(group["year"].max()),
                "oldest_publication_year": int(group["year"].min()),
                "active_year_count": len(year_counts),
                "active_year_share": safe_ratio(len(year_counts), len(year_span)),
                "semantic_coherence": mean_pairwise_text_similarity(graph, node_ids),
                "selected_density": density,
                "selected_connected_components": len(component_sizes),
                "selected_largest_component_fraction": largest_component_fraction,
                "citations_received": float(group["citations_received"].sum()),
                "mean_citation_velocity": float(group["citation_velocity"].mean()),
                "max_citation_velocity": float(group["citation_velocity"].max()),
                "mean_branch_to_trunk_score": float(
                    assignment_group["branch_to_trunk_score"].mean()
                )
                if not assignment_group.empty
                else 0.0,
                "mean_path_proximity": float(
                    assignment_group["citation_path_proximity"].mean()
                )
                if not assignment_group.empty
                else 0.0,
                "strong_assignment_count": int(
                    (assignment_group["assignment_confidence"] == "strong").sum()
                )
                if not assignment_group.empty
                else 0,
                "moderate_assignment_count": int(
                    (assignment_group["assignment_confidence"] == "moderate").sum()
                )
                if not assignment_group.empty
                else 0,
                "top_terms": aggregate_top_terms(graph, node_ids, top_terms),
                "member_node_ids": "; ".join(node_ids),
                **impact_after_removal,
            }
        )

    indicators_df = pd.DataFrame(rows)
    indicators_df = indicators_df.merge(parent_df, on="branch_label", how="left")
    return indicators_df


def load_reference_novelty(path: Path | None) -> pd.DataFrame | None:
    """Load optional branch-level reference novelty scores."""
    if path is None or not path.exists():
        return None
    novelty_df = pd.read_csv(path)
    required_columns = {
        "branch_label",
        "paper_count",
        "supported_paper_count",
        "recent_mean_reference_novelty",
        "branch_reference_novelty_score",
    }
    missing_columns = required_columns - set(novelty_df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing reference novelty columns in {path}: {sorted(missing_columns)}"
        )

    novelty_df = novelty_df.copy()
    novelty_df["reference_novelty_support_share"] = (
        novelty_df["supported_paper_count"] / novelty_df["paper_count"].clip(lower=1)
    )
    novelty_df["reference_novelty_effective_score"] = (
        novelty_df["branch_reference_novelty_score"]
        * novelty_df["reference_novelty_support_share"]
    )
    return novelty_df[
        [
            "branch_label",
            "supported_paper_count",
            "unsupported_paper_count",
            "recent_mean_reference_novelty",
            "branch_reference_novelty_score",
            "reference_novelty_support_share",
            "reference_novelty_effective_score",
        ]
    ]


def score_emergence_dimensions(
    indicators_df: pd.DataFrame,
    w_ng: float,
    w_pc: float,
    w_si: float,
    w_ur: float,
    w_rn: float,
    min_semantic_coherence: float,
    min_recent_paper_share: float,
    min_recent_growth_rate: float,
) -> pd.DataFrame:
    """Normalize and combine Xu-inspired emergence dimensions."""
    scored_df = indicators_df.copy()

    scored_df["recent_paper_share_score"] = normalize_series(scored_df["recent_paper_share"])
    scored_df["recent_growth_rate_score"] = normalize_series(scored_df["recent_growth_rate"])
    scored_df["mean_publication_year_score"] = normalize_series(
        scored_df["mean_publication_year"]
    )
    scored_df["novelty_growth_score"] = (
        (0.40 * scored_df["recent_paper_share_score"])
        + (0.40 * scored_df["recent_growth_rate_score"])
        + (0.20 * scored_df["mean_publication_year_score"])
    )

    scored_df["semantic_coherence_score"] = normalize_series(
        scored_df["semantic_coherence"]
    )
    scored_df["selected_density_score"] = normalize_series(scored_df["selected_density"])
    scored_df["active_year_share_score"] = normalize_series(scored_df["active_year_share"])
    scored_df["persistence_coherence_score"] = (
        (0.50 * scored_df["semantic_coherence_score"])
        + (0.25 * scored_df["selected_density_score"])
        + (0.25 * scored_df["active_year_share_score"])
    )

    scored_df["citation_velocity_score"] = normalize_series(
        scored_df["mean_citation_velocity"].apply(math.log1p)
    )
    scored_df["citation_count_score"] = normalize_series(
        scored_df["citations_received"].apply(math.log1p)
    )
    scored_df["trunk_connection_score"] = normalize_series(
        scored_df["mean_branch_to_trunk_score"]
    )
    scored_df["scientific_impact_score"] = (
        (0.40 * scored_df["citation_velocity_score"])
        + (0.30 * scored_df["citation_count_score"])
        + (0.30 * scored_df["trunk_connection_score"])
    )

    scored_df["parent_specificity_score"] = normalize_series(
        scored_df["parent_score_concentration"].fillna(0)
    )
    scored_df["assignment_confidence_score"] = normalize_series(
        scored_df["assignment_confidence_share"].fillna(0)
    )
    scored_df["removal_impact_score"] = normalize_series(
        scored_df["component_delta_after_branch_removal"].clip(lower=0)
        + scored_df["giant_component_loss_after_branch_removal"].clip(lower=0)
    )
    scored_df["uncertainty_reduction_score"] = (
        (0.40 * scored_df["parent_specificity_score"])
        + (0.40 * scored_df["assignment_confidence_score"])
        + (0.20 * scored_df["removal_impact_score"])
    )
    if "reference_novelty_effective_score" not in scored_df:
        scored_df["reference_novelty_effective_score"] = 0.0
    if "reference_novelty_support_share" not in scored_df:
        scored_df["reference_novelty_support_share"] = 0.0
    scored_df["reference_novelty_score"] = normalize_series(
        scored_df["reference_novelty_effective_score"].fillna(0)
    )

    scored_df["branch_emergence_score"] = (
        (w_ng * scored_df["novelty_growth_score"])
        + (w_pc * scored_df["persistence_coherence_score"])
        + (w_si * scored_df["scientific_impact_score"])
        + (w_ur * scored_df["uncertainty_reduction_score"])
        + (w_rn * scored_df["reference_novelty_score"])
    )
    scored_df["passes_semantic_guardrail"] = (
        scored_df["semantic_coherence"] >= min_semantic_coherence
    )
    scored_df["passes_recent_share_guardrail"] = (
        scored_df["recent_paper_share"] >= min_recent_paper_share
    )
    scored_df["passes_recent_growth_guardrail"] = (
        scored_df["recent_growth_rate"] >= min_recent_growth_rate
    )
    scored_df["branch_emergence_rank"] = scored_df["branch_emergence_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    scored_df["branch_emergence_type"] = scored_df.apply(classify_branch_type, axis=1)
    return scored_df.sort_values("branch_emergence_rank")


def classify_branch_type(row: pd.Series) -> str:
    """Classify branches without forcing every branch to be a trend."""
    if (
        row["novelty_growth_score"] >= 0.65
        and not row["passes_semantic_guardrail"]
    ):
        return "active_but_heterogeneous_candidate"
    if (
        row["branch_emergence_score"] >= 0.65
        and row["novelty_growth_score"] >= 0.45
        and row["persistence_coherence_score"] >= 0.45
        and row["passes_semantic_guardrail"]
        and row["passes_recent_share_guardrail"]
        and row["passes_recent_growth_guardrail"]
    ):
        return "emerging_research_direction"
    if (
        row["recent_paper_share"] <= 0.20
        and row["selected_density"] >= 0.15
        and row["semantic_coherence"] >= 0.04
    ):
        return "mature_niche_or_specialized_domain"
    if row["novelty_growth_score"] >= 0.65 and row["persistence_coherence_score"] < 0.35:
        return "active_but_heterogeneous_candidate"
    if row["scientific_impact_score"] >= 0.65 and row["novelty_growth_score"] < 0.45:
        return "established_structural_trajectory"
    return "weak_or_preliminary_candidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Xu-inspired branch emergence indicators using currently "
            "available ToS/BibFusion data."
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
    parser.add_argument("--w-ng", type=float, default=0.35)
    parser.add_argument("--w-pc", type=float, default=0.25)
    parser.add_argument("--w-si", type=float, default=0.25)
    parser.add_argument("--w-ur", type=float, default=0.15)
    parser.add_argument("--w-rn", type=float, default=0.0)
    parser.add_argument(
        "--reference-novelty",
        type=Path,
        default=None,
        help="Optional branch_reference_novelty_scores.csv file.",
    )
    parser.add_argument("--min-semantic-coherence", type=float, default=0.05)
    parser.add_argument("--min-recent-paper-share", type=float, default=0.25)
    parser.add_argument("--min-recent-growth-rate", type=float, default=0.0)
    parser.add_argument("--top-terms", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w_ng, w_pc, w_si, w_ur, w_rn = normalize_weights(
        args.w_ng,
        args.w_pc,
        args.w_si,
        args.w_ur,
        args.w_rn,
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
    raw_indicators_df = build_raw_emergence_indicators(
        graph,
        member_metrics_df,
        assignments_df,
        args.current_year,
        args.recent_window_years,
        args.top_terms,
    )
    reference_novelty_df = load_reference_novelty(args.reference_novelty)
    if reference_novelty_df is not None:
        raw_indicators_df = raw_indicators_df.merge(
            reference_novelty_df,
            on="branch_label",
            how="left",
        )
    emergence_scores_df = score_emergence_dimensions(
        raw_indicators_df,
        w_ng,
        w_pc,
        w_si,
        w_ur,
        w_rn,
        args.min_semantic_coherence,
        args.min_recent_paper_share,
        args.min_recent_growth_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    member_metrics_df.to_csv(
        args.output_dir / "branch_emergence_members.csv",
        index=False,
    )
    raw_indicators_df.to_csv(
        args.output_dir / "branch_emergence_raw_indicators.csv",
        index=False,
    )
    emergence_scores_df.to_csv(
        args.output_dir / "branch_emergence_scores.csv",
        index=False,
    )

    print(f"Branch papers: {len(member_metrics_df):,}")
    print(f"Branches: {len(emergence_scores_df):,}")
    print(
        "Weights used: "
        f"NG={w_ng:.2f}, PC={w_pc:.2f}, SI={w_si:.2f}, "
        f"UR={w_ur:.2f}, RN={w_rn:.2f}"
    )
    print(f"Saved outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
