#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3.config import ExperimentConfig, load_experiment_config
from hw3.utils import ensure_parent_dir, resolve_path


def _resolve_config_path(path_text: str) -> Path:
    config_path = Path(path_text)
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    return config_path


def _load_suite_configs(suite_path: Path) -> list[tuple[Path, list[str]]]:
    with suite_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in suite: {suite_path}")
    config_paths: list[tuple[Path, list[str]]] = []
    for experiment in experiments:
        config_path = Path(experiment["config"])
        if not config_path.is_absolute():
            config_path = (suite_path.parent / config_path).resolve()
        config_paths.append((config_path, experiment.get("overrides", [])))
    return config_paths


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_metrics(config: ExperimentConfig) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    output_dir = resolve_path(config.training.output_dir)
    metrics_path = resolve_path(config.evaluation.metrics_path)
    training_metrics_path = output_dir / "training_metrics.json"
    return _read_json(metrics_path), _read_json(training_metrics_path)


def _build_row(config_path: Path, config: ExperimentConfig) -> dict[str, Any]:
    eval_metrics, training_metrics = _load_metrics(config)
    quantization_mode = "4bit" if config.quantization.enabled and config.quantization.load_in_4bit else "none"
    status = "completed" if eval_metrics is not None else "not_run"
    row: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "config_path": str(config_path.relative_to(ROOT)),
        "output_dir": config.training.output_dir,
        "model": config.model.name_or_path,
        "adapter": config.adapter.type,
        "quantization": quantization_mode,
        "rank": config.adapter.rank,
        "alpha": config.adapter.alpha,
        "dropout": config.adapter.dropout,
        "epochs": config.training.num_train_epochs,
        "learning_rate": config.training.learning_rate,
        "max_length": config.model.max_length,
        "per_device_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "effective_batch_without_ddp": config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps,
        "status": status,
        "accuracy": eval_metrics.get("accuracy") if eval_metrics else None,
        "correct": eval_metrics.get("correct") if eval_metrics else None,
        "total": eval_metrics.get("total") if eval_metrics else None,
        "train_loss": training_metrics.get("train_loss") if training_metrics else None,
        "train_runtime": training_metrics.get("train_runtime") if training_metrics else None,
        "train_steps_per_second": training_metrics.get("train_steps_per_second") if training_metrics else None,
        "results_path": eval_metrics.get("results_path") if eval_metrics else None,
    }
    return row


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: ((row["accuracy"] is None), -(row["accuracy"] or -1.0), row["experiment_name"]),
    )


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    lines: list[str] = ["# Experiment Summary", ""]
    completed = [row for row in rows if row["accuracy"] is not None]
    if completed:
        best = max(completed, key=lambda row: row["accuracy"])
        lines.extend(
            [
                "## Best Result",
                "",
                f"- Experiment: `{best['experiment_name']}`",
                f"- Config: `{best['config_path']}`",
                f"- Accuracy: `{best['accuracy']:.4f}`",
                f"- Model: `{best['model']}`",
                f"- Adapter: `{best['adapter']}`",
                f"- Quantization: `{best['quantization']}`",
                "",
            ]
        )
    else:
        lines.extend(["## Best Result", "", "No completed evaluation metrics were found yet.", ""])

    lines.extend(
        [
            "## Table",
            "",
            "| Experiment | Model | Adapter | Quantization | Accuracy | Epochs | Batch | Grad Accum | Results | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        accuracy = "-" if row["accuracy"] is None else f"{row['accuracy']:.4f}"
        results_path = row["results_path"] or "-"
        lines.append(
            f"| `{row['experiment_name']}` | `{row['model']}` | `{row['adapter']}` | `{row['quantization']}` | {accuracy} | {row['epochs']} | {row['per_device_batch_size']} | {row['gradient_accumulation_steps']} | `{results_path}` | `{row['status']}` |"
        )
    lines.append("")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _default_output_prefix(
    first_config_path: Path,
    rows: list[dict[str, Any]],
    suite_path: Path | None = None,
) -> Path:
    if suite_path is not None:
        return ROOT / "reports" / f"{suite_path.stem}_summary"
    if len(rows) == 1:
        return ROOT / "reports" / f"{rows[0]['experiment_name']}_summary"
    return ROOT / "reports" / f"{first_config_path.stem}_summary"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation/training metrics across configs.")
    parser.add_argument("--config", action="append", default=[], help="Experiment config path. Can be repeated.")
    parser.add_argument("--suite", default=None, help="Optional suite YAML path.")
    parser.add_argument("--override", action="append", default=[], help="Optional dotted-path overrides applied to direct --config inputs.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix without extension. Defaults to reports/<name>_summary")
    args = parser.parse_args()

    config_specs: list[tuple[Path, list[str]]] = []
    resolved_suite_path: Path | None = None
    if args.suite is not None:
        suite_path = Path(args.suite)
        if not suite_path.is_absolute():
            suite_path = (ROOT / suite_path).resolve()
        resolved_suite_path = suite_path
        config_specs.extend(_load_suite_configs(suite_path))
    config_specs.extend((_resolve_config_path(config_text), args.override) for config_text in args.config)

    if not config_specs:
        raise ValueError("Provide at least one --config or a --suite.")

    rows = _sort_rows([
        _build_row(config_path, load_experiment_config(config_path, overrides=overrides or None))
        for config_path, overrides in config_specs
    ])
    output_prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix
        else _default_output_prefix(config_specs[0][0], rows, suite_path=resolved_suite_path)
    )
    _write_json(Path(f"{output_prefix}.json"), rows)
    _write_csv(Path(f"{output_prefix}.csv"), rows)
    _write_markdown(Path(f"{output_prefix}.md"), rows)
    print(f"Wrote summary to {output_prefix}.json/.csv/.md")


if __name__ == "__main__":
    main()
