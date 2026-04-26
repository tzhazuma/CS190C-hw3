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
SUMMARY_SCRIPT = ROOT / "scripts" / "summarize_results.py"


def _resolve_config_path(suite_file: Path, config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    return (suite_file.parent / path).resolve()


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _build_train_command(
    python_bin: str,
    config_path: Path,
    overrides: list[str],
    *,
    num_gpus: int,
    master_port: int,
) -> list[str]:
    if num_gpus > 1:
        command = [
            python_bin,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_gpus),
            "--master_port",
            str(master_port),
            str(TRAIN_SCRIPT),
            "--config",
            str(config_path),
        ]
    else:
        command = [python_bin, str(TRAIN_SCRIPT), "--config", str(config_path)]

    for override in overrides:
        command.extend(["--override", override])
    return command


def _build_eval_command(python_bin: str, config_path: Path, overrides: list[str]) -> list[str]:
    command = [python_bin, str(EVAL_SCRIPT), "--config", str(config_path)]
    for override in overrides:
        command.extend(["--override", override])
    return command


def _build_summary_command(python_bin: str, config_path: Path, overrides: list[str]) -> list[str]:
    command = [python_bin, str(SUMMARY_SCRIPT), "--config", str(config_path)]
    for override in overrides:
        command.extend(["--override", override])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequence of training + evaluation experiments.")
    parser.add_argument("--suite", required=True, help="Path to a YAML suite file.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation stage.")
    parser.add_argument("--skip-summary", action="store_true", help="Skip the summary stage.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use for training each experiment.")
    parser.add_argument("--master-port-base", type=int, default=29500, help="Base port used for torch.distributed runs.")
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    with suite_path.open("r", encoding="utf-8") as handle:
        suite_payload = yaml.safe_load(handle) or {}

    experiments = suite_payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in suite: {suite_path}")

    python_bin = sys.executable
    for index, experiment in enumerate(experiments):
        config_path = _resolve_config_path(suite_path, experiment["config"])
        overrides = experiment.get("overrides", [])
        if not args.skip_train:
            command = _build_train_command(
                python_bin,
                config_path,
                overrides,
                num_gpus=args.num_gpus,
                master_port=args.master_port_base + index,
            )
            _run_command(command)
        if not args.skip_eval:
            command = _build_eval_command(python_bin, config_path, overrides)
            _run_command(command)
        if not args.skip_summary:
            command = _build_summary_command(python_bin, config_path, overrides)
            _run_command(command)


if __name__ == "__main__":
    main()
