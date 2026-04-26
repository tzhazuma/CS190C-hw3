from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hw3.utils import count_parameters, ensure_dir, write_json


def is_linear_like(module: nn.Module) -> bool:
    return all(hasattr(module, attribute) for attribute in ("in_features", "out_features", "weight"))


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        weight_dtype = getattr(base_layer.weight, "dtype", torch.float32)
        if weight_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            weight_dtype = torch.float32
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=weight_dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=weight_dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        lora_input = self.dropout(inputs).to(self.lora_A.dtype)
        lora_hidden = F.linear(lora_input, self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_output + lora_output.to(base_output.dtype)


def _module_matches(module_name: str, target_modules: list[str]) -> bool:
    leaf_name = module_name.split(".")[-1]
    return leaf_name in target_modules or any(module_name.endswith(target) for target in target_modules)


def _split_module_name(module_name: str) -> tuple[str, str]:
    if "." not in module_name:
        return "", module_name
    parent_name, child_name = module_name.rsplit(".", maxsplit=1)
    return parent_name, child_name


def mark_only_manual_lora_as_trainable(
    model: nn.Module,
    *,
    bias: str = "none",
    modules_to_save: list[str] | None = None,
) -> None:
    save_modules = set(modules_to_save or [])
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
            if bias == "lora_only" and getattr(module.base_layer, "bias", None) is not None:
                module.base_layer.bias.requires_grad = True

    if bias == "all":
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.requires_grad = True

    if save_modules:
        for name, parameter in model.named_parameters():
            if any(token in name for token in save_modules):
                parameter.requires_grad = True


def inject_manual_lora(
    model: nn.Module,
    *,
    target_modules: list[str],
    rank: int,
    alpha: int,
    dropout: float,
    bias: str = "none",
    freeze_base_model: bool = True,
    modules_to_save: list[str] | None = None,
) -> list[str]:
    replaced_modules: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if isinstance(module, LoRALinear):
            continue
        if not is_linear_like(module):
            continue
        if not _module_matches(module_name, target_modules):
            continue

        parent_name, child_name = _split_module_name(module_name)
        parent_module = model.get_submodule(parent_name) if parent_name else model
        setattr(parent_module, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced_modules.append(module_name)

    if freeze_base_model:
        mark_only_manual_lora_as_trainable(model, bias=bias, modules_to_save=modules_to_save)

    return replaced_modules


def manual_lora_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_manual_lora_adapter(model: nn.Module, output_dir: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = ensure_dir(output_dir)
    state_dict = manual_lora_trainable_state_dict(model)
    adapter_path = output_path / "manual_lora_adapter.pt"
    torch.save({"state_dict": state_dict, "metadata": metadata}, adapter_path)
    summary = dict(metadata)
    summary.update(count_parameters(model))
    write_json(output_path / "manual_lora_metadata.json", summary)
    return adapter_path


def load_manual_lora_adapter(model: nn.Module, adapter_path: str | Path) -> dict[str, list[str]]:
    adapter_file = Path(adapter_path)
    if adapter_file.is_dir():
        adapter_file = adapter_file / "manual_lora_adapter.pt"
    payload = torch.load(adapter_file, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
