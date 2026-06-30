from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )


def run_stage(script_names: list[str], dry_run: bool = False) -> None:
    for script_name in script_names:
        script_path = REPO_ROOT / "src" / script_name
        command = [sys.executable, str(script_path)]
        printable = " ".join(command)
        if dry_run:
            print(printable)
            continue
        print(f"\n>>> {printable}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)

