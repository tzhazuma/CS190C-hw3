#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3.config import load_experiment_config
from hw3.evaluation import evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained experiment on gsm8k_val.jsonl.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--adapter-path", default=None, help="Optional adapter checkpoint override.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted-path override, for example generation.max_new_tokens=512",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()

    config = load_experiment_config(config_path, overrides=args.override)
    metrics = evaluate_experiment(config, adapter_path=args.adapter_path)
    print(metrics)


if __name__ == "__main__":
    main()
