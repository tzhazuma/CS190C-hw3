from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    root = Path(base_dir) if base_dir is not None else PROJECT_ROOT
    return (root / path_obj).resolve()


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_parent_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path_obj = ensure_parent_dir(path)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path_obj = ensure_parent_dir(path)
    with path_obj.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    name = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[name]


def get_default_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def count_parameters(model: torch.nn.Module) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        param_count = parameter.numel()
        total += param_count
        if parameter.requires_grad:
            trainable += param_count
    ratio = 0.0 if total == 0 else trainable / total
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": ratio,
    }
