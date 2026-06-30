from __future__ import annotations

import argparse

from pipeline_utils import add_stage_arguments, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 02: build the directed citation network. The current graph builder "
            "also applies SAP and writes ToS node attributes."
        )
    )
    add_stage_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_stage(["build_citation_network.py"], dry_run=args.dry_run, config_path=args.config)


if __name__ == "__main__":
    main()
