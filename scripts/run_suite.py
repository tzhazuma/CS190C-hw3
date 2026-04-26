#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train.py"
EVAL_SCRIPT = ROOT / "scripts" / "evaluate.py"


def _resolve_config_path(suite_file: Path, config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (suite_file.parent / path).resolve()


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequence of training + evaluation experiments.")
    parser.add_argument("--suite", required=True, help="Path to a YAML suite file.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation stage.")
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    with suite_path.open("r", encoding="utf-8") as handle:
        suite_payload = yaml.safe_load(handle) or {}

    experiments = suite_payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in suite: {suite_path}")

    python_bin = sys.executable
    for experiment in experiments:
        config_path = _resolve_config_path(suite_path, experiment["config"])
        overrides = experiment.get("overrides", [])
        if not args.skip_train:
            command = [python_bin, str(TRAIN_SCRIPT), "--config", str(config_path)]
            for override in overrides:
                command.extend(["--override", override])
            _run_command(command)
        if not args.skip_eval:
            command = [python_bin, str(EVAL_SCRIPT), "--config", str(config_path)]
            for override in overrides:
                command.extend(["--override", override])
            _run_command(command)


if __name__ == "__main__":
    main()
