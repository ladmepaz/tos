from __future__ import annotations

import argparse

from pipeline_utils import add_dry_run_argument, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 02: build the directed citation network. The current graph builder "
            "also applies SAP and writes ToS node attributes."
        )
    )
    add_dry_run_argument(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(["build_citation_network.py"], dry_run=args.dry_run)


if __name__ == "__main__":
    main()

