from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 04: compute root similarities, clusters, and visualization metrics."
    )
    add_dry_run_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(
        [
            "root_tfidf_similarity.py",
            "root_cocitation_similarity.py",
            "root_structural_similarity.py",
            "root_combined_similarity.py",
            "root_cluster_metrics.py",
            "root_visualization_metrics.py",
            "export_root_visualization_svg.py",
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

