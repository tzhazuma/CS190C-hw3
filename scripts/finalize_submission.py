#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hw3.config import ExperimentConfig, load_experiment_config
from hw3.utils import ensure_dir, ensure_parent_dir, resolve_path


DEFAULT_STUDENT_NAME = "tangzhihao"
DEFAULT_STUDENT_ID = "2022533131"


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

    specs: list[tuple[Path, list[str]]] = []
    for experiment in experiments:
        config_path = Path(experiment["config"])
        if not config_path.is_absolute():
            config_path = (suite_path.parent / config_path).resolve()
        specs.append((config_path, experiment.get("overrides", [])))
    return specs


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _query_gpu_summary() -> tuple[str, str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown", "unknown"

    if not output:
        return "unknown", "unknown"

    rows = [line.strip() for line in output.splitlines() if line.strip()]
    names = [row.split(",")[0].strip() for row in rows]
    memories = [row.split(",", maxsplit=1)[1].strip() for row in rows if "," in row]
    gpu_summary = f"{len(rows)} x {names[0]}" if len(set(names)) == 1 else "; ".join(rows)
    vram_summary = ", ".join(memories) if memories else "unknown"
    return gpu_summary, vram_summary


def _query_cuda_version() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return "unknown"
    for line in output.splitlines():
        if "CUDA Version:" in line:
            return line.split("CUDA Version:", maxsplit=1)[1].split()[0]
    return "unknown"


def _build_hardware_info() -> dict[str, str]:
    gpu_summary, vram_summary = _query_gpu_summary()
    return {
        "gpu_summary": os.getenv("SUBMISSION_GPU_SUMMARY", gpu_summary),
        "vram_summary": os.getenv("SUBMISSION_VRAM_SUMMARY", vram_summary),
        "cuda_version": os.getenv("SUBMISSION_CUDA_VERSION", _query_cuda_version()),
        "python_version": platform.python_version(),
        "torch_version": _safe_version("torch"),
        "transformers_version": _safe_version("transformers"),
        "peft_version": _safe_version("peft"),
    }


def _load_run_record(config_path: Path, overrides: list[str]) -> dict[str, Any]:
    config = load_experiment_config(config_path, overrides=overrides or None)
    output_dir = resolve_path(config.training.output_dir)
    metrics_path = resolve_path(config.evaluation.metrics_path)
    results_path = resolve_path(config.evaluation.results_path)
    training_metrics_path = output_dir / "training_metrics.json"
    eval_metrics = _read_json(metrics_path)
    training_metrics = _read_json(training_metrics_path)

    return {
        "config_path": config_path,
        "overrides": overrides,
        "config": config,
        "output_dir": output_dir,
        "eval_metrics": eval_metrics,
        "training_metrics": training_metrics,
        "results_path": results_path,
        "accuracy": eval_metrics.get("accuracy") if eval_metrics else None,
        "correct": eval_metrics.get("correct") if eval_metrics else None,
        "total": eval_metrics.get("total") if eval_metrics else None,
    }


def _completed_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["accuracy"] is not None and record["results_path"].exists()]


def _select_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = _completed_records(records)
    if not completed:
        raise ValueError("No completed experiment with metrics.json and results.jsonl was found.")
    return sorted(completed, key=lambda record: (-record["accuracy"], record["config"].experiment_name))[0]


def _format_quantization(config: ExperimentConfig) -> str:
    if not config.quantization.enabled:
        return "off"
    if config.quantization.load_in_4bit:
        return "4-bit"
    if config.quantization.load_in_8bit:
        return "8-bit"
    return "on"


def _format_accuracy(record: dict[str, Any]) -> str:
    if record["accuracy"] is None:
        return "not-run"
    return f"{record['accuracy'] * 100:.2f}%"


def _build_experiments_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Experiment | Model | Adapter | Quantization | Accuracy | Output |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in sorted(records, key=lambda item: ((item["accuracy"] is None), -(item["accuracy"] or -1.0), item["config"].experiment_name)):
        config = record["config"]
        accuracy = "-" if record["accuracy"] is None else f"{record['accuracy'] * 100:.2f}%"
        lines.append(
            f"| `{config.experiment_name}` | `{config.model.name_or_path}` | `{config.adapter.type}` | `{_format_quantization(config)}` | {accuracy} | `{config.training.output_dir}` |"
        )
    return "\n".join(lines)


def _render_template(template_path: Path, values: dict[str, str]) -> str:
    content = template_path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace(f"{{{{{key}}}}}", value)
    return content


def _write_text(path: Path, content: str) -> None:
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        config = record["config"]
        rows.append(
            {
                "experiment_name": config.experiment_name,
                "config_path": str(record["config_path"].relative_to(ROOT)),
                "model": config.model.name_or_path,
                "adapter": config.adapter.type,
                "quantization": _format_quantization(config),
                "accuracy": record["accuracy"],
                "correct": record["correct"],
                "total": record["total"],
                "output_dir": config.training.output_dir,
                "results_path": str(record["results_path"]) if record["results_path"].exists() else "",
            }
        )
    return rows


def _copy_best_run_artifacts(best_record: dict[str, Any], submission_dir: Path) -> dict[str, Path]:
    output_dir = best_record["output_dir"]
    copied: dict[str, Path] = {}

    results_dst = submission_dir / "results.jsonl"
    shutil.copy2(best_record["results_path"], results_dst)
    copied["results"] = results_dst

    metrics_src = resolve_path(best_record["config"].evaluation.metrics_path)
    if metrics_src and metrics_src.exists():
        metrics_dst = submission_dir / "best_run" / "metrics.json"
        ensure_parent_dir(metrics_dst)
        shutil.copy2(metrics_src, metrics_dst)
        copied["metrics"] = metrics_dst

    training_metrics_src = output_dir / "training_metrics.json"
    if training_metrics_src.exists():
        training_metrics_dst = submission_dir / "best_run" / "training_metrics.json"
        ensure_parent_dir(training_metrics_dst)
        shutil.copy2(training_metrics_src, training_metrics_dst)
        copied["training_metrics"] = training_metrics_dst

    resolved_config_src = output_dir / "config.resolved.yaml"
    if resolved_config_src.exists():
        resolved_config_dst = submission_dir / "best_run" / "config.resolved.yaml"
        ensure_parent_dir(resolved_config_dst)
        shutil.copy2(resolved_config_src, resolved_config_dst)
        copied["resolved_config"] = resolved_config_dst

    return copied


def _repo_snapshot_files() -> list[Path]:
    files: list[Path] = []
    for relative_path in [
        Path("Assignment_3.md"),
        Path("requirements.txt"),
        Path(".gitignore"),
        Path("gsm8k_val.jsonl"),
    ]:
        absolute = ROOT / relative_path
        if absolute.exists():
            files.append(absolute)

    for directory_name in ["src", "scripts", "configs", "docs"]:
        directory = ROOT / directory_name
        if not directory.exists():
            continue
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                files.append(file_path)
    return files


def _write_zip(zip_path: Path, submission_dir: Path, best_record: dict[str, Any], copied_artifacts: dict[str, Path]) -> None:
    ensure_parent_dir(zip_path)
    snapshot_files = _repo_snapshot_files()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_dir / "README.md", arcname="README.md")
        archive.write(submission_dir / "final_report.md", arcname="final_report.md")
        archive.write(submission_dir / "results.jsonl", arcname="results.jsonl")
        archive.write(submission_dir / "submission_manifest.json", arcname="submission_manifest.json")
        archive.write(submission_dir / "experiment_summary.md", arcname="experiment_summary.md")
        archive.write(submission_dir / "experiment_summary.csv", arcname="experiment_summary.csv")
        archive.write(submission_dir / "experiment_summary.json", arcname="experiment_summary.json")

        for key, file_path in copied_artifacts.items():
            if key == "results":
                continue
            archive.write(file_path, arcname=str(file_path.relative_to(submission_dir)))

        for file_path in snapshot_files:
            relative = file_path.relative_to(ROOT)
            if relative == Path("README.md"):
                continue
            archive.write(file_path, arcname=str(relative))


def _template_values(best_record: dict[str, Any], records: list[dict[str, Any]], hardware: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    config = best_record["config"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    best_run_command = f"bash scripts/run_train_eval.sh {best_record['config_path'].relative_to(ROOT)}"
    return {
        "generated_at": generated_at,
        "student_name": args.student_name,
        "student_id": args.student_id,
        "course_name": args.course_name,
        "best_experiment_name": config.experiment_name,
        "best_config_path": str(best_record["config_path"].relative_to(ROOT)),
        "base_model": config.model.name_or_path,
        "adapter_type": config.adapter.type,
        "quantization": _format_quantization(config),
        "accuracy_percent": _format_accuracy(best_record),
        "correct": str(best_record["correct"]),
        "total": str(best_record["total"]),
        "output_dir": config.training.output_dir,
        "rank": str(config.adapter.rank),
        "alpha": str(config.adapter.alpha),
        "dropout": str(config.adapter.dropout),
        "learning_rate": str(config.training.learning_rate),
        "epochs": str(config.training.num_train_epochs),
        "batch_size": str(config.training.per_device_train_batch_size),
        "gradient_accumulation": str(config.training.gradient_accumulation_steps),
        "max_length": str(config.model.max_length),
        "gpu_summary": f"{hardware['gpu_summary']} (VRAM: {hardware['vram_summary']})",
        "cuda_version": hardware["cuda_version"],
        "python_version": hardware["python_version"],
        "torch_version": hardware["torch_version"],
        "transformers_version": hardware["transformers_version"],
        "peft_version": hardware["peft_version"],
        "results_source_path": str(best_record["results_path"]),
        "experiments_table": _build_experiments_table(records),
        "target_modules": ", ".join(config.adapter.target_modules),
        "best_run_command": best_run_command,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best run, fill submission templates, and create a submission zip.")
    parser.add_argument("--config", action="append", default=[], help="Experiment config path. Can be repeated.")
    parser.add_argument("--suite", default=None, help="Optional suite YAML path.")
    parser.add_argument("--override", action="append", default=[], help="Optional overrides applied to direct --config arguments.")
    parser.add_argument("--submission-dir", default=os.getenv("SUBMISSION_DIR", "submission"), help="Directory for generated submission artifacts.")
    parser.add_argument("--zip-name", default=os.getenv("SUBMISSION_ZIP_NAME", "CS190C-hw3-submission.zip"), help="Submission zip filename.")
    parser.add_argument("--student-name", default=os.getenv("STUDENT_NAME", DEFAULT_STUDENT_NAME), help="Student name for autofilled templates.")
    parser.add_argument("--student-id", default=os.getenv("STUDENT_ID", DEFAULT_STUDENT_ID), help="Student ID for autofilled templates.")
    parser.add_argument("--course-name", default=os.getenv("COURSE_NAME", "CS190C"), help="Course name for autofilled templates.")
    parser.add_argument("--skip-root-readme", action="store_true", help="Do not update the repository root README final results section.")
    return parser.parse_args()


def _autofill_root_readme(best_record: dict[str, Any]) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "autofill_root_readme.py"),
        "--config",
        str(best_record["config_path"]),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    args = _parse_args()
    config_specs: list[tuple[Path, list[str]]] = []
    if args.suite is not None:
        suite_path = _resolve_config_path(args.suite)
        config_specs.extend(_load_suite_configs(suite_path))
    config_specs.extend((_resolve_config_path(config_text), args.override) for config_text in args.config)

    if not config_specs:
        raise ValueError("Provide at least one --config or a --suite.")

    records = [_load_run_record(config_path, overrides) for config_path, overrides in config_specs]
    best_record = _select_best_record(records)
    hardware = _build_hardware_info()
    values = _template_values(best_record, records, hardware, args)

    submission_dir = ensure_dir(resolve_path(args.submission_dir))
    summary_rows = _summary_rows(records)

    summary_json_path = submission_dir / "experiment_summary.json"
    summary_csv_path = submission_dir / "experiment_summary.csv"
    summary_md_path = submission_dir / "experiment_summary.md"
    _write_json(summary_json_path, summary_rows)
    _write_csv(summary_csv_path, summary_rows)
    _write_text(summary_md_path, "# Experiment Summary\n\n" + values["experiments_table"] + "\n")

    readme_template = ROOT / "docs" / "templates" / "submission_readme_template.md"
    report_template = ROOT / "docs" / "templates" / "final_report_autofill_template.md"
    readme_output = submission_dir / "README.md"
    report_output = submission_dir / "final_report.md"
    _write_text(readme_output, _render_template(readme_template, values))
    _write_text(report_output, _render_template(report_template, values))

    copied_artifacts = _copy_best_run_artifacts(best_record, submission_dir)
    manifest = {
        "best_experiment_name": best_record["config"].experiment_name,
        "best_config_path": str(best_record["config_path"].relative_to(ROOT)),
        "accuracy": best_record["accuracy"],
        "correct": best_record["correct"],
        "total": best_record["total"],
        "results_source_path": str(best_record["results_path"]),
        "submission_dir": str(submission_dir),
        "zip_name": args.zip_name,
    }
    _write_json(submission_dir / "submission_manifest.json", manifest)

    if not args.skip_root_readme:
        _autofill_root_readme(best_record)

    zip_path = submission_dir / args.zip_name
    _write_zip(zip_path, submission_dir, best_record, copied_artifacts)
    print(f"Best experiment: {best_record['config'].experiment_name}")
    print(f"Submission package: {zip_path}")


if __name__ == "__main__":
    main()
