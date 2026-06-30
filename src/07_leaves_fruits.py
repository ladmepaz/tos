from __future__ import annotations

import argparse

from pipeline_utils import add_stage_arguments, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 07: identify leaf-like frontier papers and fruit candidates."
    )
    add_stage_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(
        [
            "export_leaf_like_nodes_from_gexf.py",
            "fruit_candidates.py",
            "export_fruit_visualization_svg.py",
            "export_leaf_like_canopy_visualization_svg.py",
        ],
        dry_run=args.dry_run,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
