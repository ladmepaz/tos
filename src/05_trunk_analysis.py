from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 05: compute trunk subtopics and trunk visualization metrics."
    )
    add_dry_run_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(
        [
            "trunk_combined_similarity.py",
            "trunk_visualization_metrics.py",
            "export_trunk_visualization_svg.py",
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

