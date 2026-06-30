from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline_utils import DEFAULT_CONFIG, REPO_ROOT


STAGES = [
    (1, "01_tidy_data.py", "Clean BibFusion citation and article files"),
    (2, "02_build_citation_network.py", "Build citation network and apply SAP labels"),
    (3, "03_apply_sap.py", "Summarize ToS/SAP labels"),
    (4, "04_roots_analysis.py", "Analyze roots"),
    (5, "05_trunk_analysis.py", "Analyze trunk"),
    (6, "06_branches_analysis.py", "Analyze branches"),
    (7, "07_leaves_fruits.py", "Analyze leaves and fruits"),
    (8, "08_export_visualizations.py", "Export final visualizations"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete Tree of Science 3 reproducibility pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the ToS 3 JSON configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stage commands without running them.",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        choices=range(1, 9),
        default=1,
        metavar="{1..8}",
        help="First numbered stage to run.",
    )
    parser.add_argument(
        "--end-stage",
        type=int,
        choices=range(1, 9),
        default=8,
        metavar="{1..8}",
        help="Last numbered stage to run.",
    )
    return parser.parse_args()


def selected_stages(start_stage: int, end_stage: int) -> list[tuple[int, str, str]]:
    if start_stage > end_stage:
        raise SystemExit("--start-stage must be less than or equal to --end-stage.")
    return [
        stage
        for stage in STAGES
        if start_stage <= stage[0] <= end_stage
    ]


def main() -> None:
    args = parse_args()
    stages = selected_stages(args.start_stage, args.end_stage)

    print("Tree of Science 3 pipeline")
    print(f"Repository: {REPO_ROOT}")
    print(f"Config: {args.config}")
    print(f"Stages: {stages[0][0]} to {stages[-1][0]}")

    for stage_number, script_name, description in stages:
        script_path = REPO_ROOT / "src" / script_name
        command = [
            sys.executable,
            str(script_path),
            "--config",
            str(args.config),
        ]
        printable = " ".join(command)
        print(f"\n[{stage_number}] {description}")
        if args.dry_run:
            print(printable)
            continue
        print(f">>> {printable}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        print("\nDry run complete. No files were changed.")
    else:
        print("\nPipeline complete.")


if __name__ == "__main__":
    main()

