#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3.config import load_experiment_config
from hw3.utils import resolve_path


FINAL_RESULTS_HEADER = "## Final Results Section"
DELIVERABLES_HEADER = "## Deliverables Checklist"


def _resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_best_context(config_path: Path) -> dict[str, str]:
    config = load_experiment_config(config_path)
    metrics_path = resolve_path(config.evaluation.metrics_path)
    if metrics_path is None or not metrics_path.exists():
        raise ValueError(f"Metrics file not found for config: {config_path}")

    import json

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    accuracy = float(metrics["accuracy"])
    accuracy_percent = f"{accuracy * 100:.2f}% ({metrics['correct']}/{metrics['total']})"
    hardware = "${GPU info from submission/final_report.md}" if False else "See submission/final_report.md"

    return {
        "best_config": str(config_path.relative_to(ROOT)),
        "accuracy": accuracy_percent,
        "base_model": config.model.name_or_path,
        "adapter_type": config.adapter.type,
        "quantization": "off" if not config.quantization.enabled else ("4-bit" if config.quantization.load_in_4bit else "8-bit" if config.quantization.load_in_8bit else "on"),
        "rank": str(config.adapter.rank),
        "alpha": str(config.adapter.alpha),
        "epochs": str(config.training.num_train_epochs),
        "batch_size": str(config.training.per_device_train_batch_size),
        "gradient_accumulation": str(config.training.gradient_accumulation_steps),
        "learning_rate": str(config.training.learning_rate),
        "hardware": hardware,
    }


def _replace_final_results_section(readme_text: str, values: dict[str, str]) -> str:
    if FINAL_RESULTS_HEADER not in readme_text or DELIVERABLES_HEADER not in readme_text:
        raise ValueError("README.md does not contain the expected Final Results Section markers.")

    start = readme_text.index(FINAL_RESULTS_HEADER)
    end = readme_text.index(DELIVERABLES_HEADER)
    replacement = (
        f"{FINAL_RESULTS_HEADER}\n\n"
        f"Autofilled by `scripts/autofill_root_readme.py` after selecting the best completed run.\n\n"
        f"| Item | Value |\n"
        f"| --- | --- |\n"
        f"| Best config | `{values['best_config']}` |\n"
        f"| Final validation accuracy | {values['accuracy']} |\n"
        f"| Base model | `{values['base_model']}` |\n"
        f"| Adapter type | `{values['adapter_type']}` |\n"
        f"| Quantization | `{values['quantization']}` |\n"
        f"| LoRA rank | {values['rank']} |\n"
        f"| LoRA alpha | {values['alpha']} |\n"
        f"| Epochs | {values['epochs']} |\n"
        f"| Batch size | {values['batch_size']} |\n"
        f"| Gradient accumulation | {values['gradient_accumulation']} |\n"
        f"| Learning rate | {values['learning_rate']} |\n"
        f"| Hardware used | {values['hardware']} |\n\n"
    )
    return readme_text[:start] + replacement + readme_text[end:]


def _replace_deliverables_checklist(readme_text: str) -> str:
    replacements = {
        "- [ ] final best `results.jsonl`": "- [x] final best `results.jsonl`",
        "- [ ] final measured accuracy": "- [x] final measured accuracy",
    }
    updated = readme_text
    for old, new in replacements.items():
        updated = updated.replace(old, new)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Autofill the repository root README Final Results Section.")
    parser.add_argument("--config", required=True, help="Best experiment config path.")
    parser.add_argument("--readme", default=str(ROOT / "README.md"), help="README file to update.")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    readme_path = Path(args.readme)
    if not readme_path.is_absolute():
        readme_path = (ROOT / readme_path).resolve()

    values = _load_best_context(config_path)
    original = _read_text(readme_path)
    updated = _replace_final_results_section(original, values)
    updated = _replace_deliverables_checklist(updated)
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated README final results using {config_path}")


if __name__ == "__main__":
    main()
