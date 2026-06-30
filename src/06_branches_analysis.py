from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 06: identify branch communities, core papers, and SVG outputs."
    )
    add_dry_run_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(
        [
            "branch_leiden_comparison.py",
            "branch_trunk_assignment.py",
            "branch_member_roles.py",
            "branch_visualization_metrics.py",
            "export_branch_visualization_svg.py",
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

