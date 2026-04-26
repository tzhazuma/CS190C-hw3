#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3.config import load_experiment_config
from hw3.training import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GSM8K fine-tuning experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted-path override, for example training.learning_rate=1e-4",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()

    config = load_experiment_config(config_path, overrides=args.override)
    metrics = train_experiment(config)
    print(metrics)


if __name__ == "__main__":
    main()
