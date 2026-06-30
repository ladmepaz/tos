from __future__ import annotations

import argparse

from pipeline_utils import add_stage_arguments, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 05: compute trunk subtopics and trunk visualization metrics."
    )
    add_stage_arguments(parser)
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
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
