from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 01: clean BibFusion citation and article files."
    )
    add_dry_run_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(
        [
            "tidy_citations.py",
            "tidy_articles.py",
        ],
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

