from __future__ import annotations

import argparse

from pipeline_utils import add_stage_arguments, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 06: identify branch communities, core papers, and SVG outputs."
    )
    add_stage_arguments(parser)
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
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
