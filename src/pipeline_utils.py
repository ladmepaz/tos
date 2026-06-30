from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "tos3_config.json"


def add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the ToS 3 JSON configuration file.",
    )


def add_stage_arguments(parser: argparse.ArgumentParser) -> None:
    add_dry_run_argument(parser)
    add_config_argument(parser)


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    resolved_path = config_path
    if not resolved_path.is_absolute():
        resolved_path = REPO_ROOT / resolved_path
    if not resolved_path.exists():
        return {}
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def format_cli_args(arguments: dict[str, Any]) -> list[str]:
    cli_args: list[str] = []
    for key, value in arguments.items():
        if value is None or value is False:
            continue
        option = key if str(key).startswith("--") else f"--{key}"
        if value is True:
            cli_args.append(option)
            continue
        cli_args.extend([option, str(value)])
    return cli_args


def configured_args(config: dict[str, Any], script_name: str) -> list[str]:
    script_arguments = config.get("script_arguments", {})
    arguments = script_arguments.get(script_name, {})
    if not isinstance(arguments, dict):
        raise TypeError(f"Config script arguments for {script_name} must be an object.")
    return format_cli_args(arguments)


def run_stage(
    script_names: list[str],
    dry_run: bool = False,
    config_path: Path | None = DEFAULT_CONFIG,
) -> None:
    config = load_config(config_path)
    for script_name in script_names:
        script_path = REPO_ROOT / "src" / script_name
        command = [sys.executable, str(script_path), *configured_args(config, script_name)]
        printable = " ".join(command)
        if dry_run:
            print(printable)
            continue
        print(f"\n>>> {printable}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)
