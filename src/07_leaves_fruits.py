from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 07: identify leaf-like frontier papers and fruit candidates."
    )
    add_dry_run_argument(parser)
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
    )


if __name__ == "__main__":
    main()

