from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hw3.config import ExperimentConfig
from hw3.manual_lora import inject_manual_lora, load_manual_lora_adapter
from hw3.utils import count_parameters, resolve_path, resolve_torch_dtype


def _resolve_device_map(config: ExperimentConfig) -> str | dict[str, int] | None:
    if config.model.device_map:
        return config.model.device_map
    if not config.quantization.enabled:
        return None
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return {"": local_rank}
    return "auto"


def build_quantization_config(config: ExperimentConfig) -> BitsAndBytesConfig | None:
    if not config.quantization.enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        load_in_8bit=config.quantization.load_in_8bit,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=resolve_torch_dtype(config.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
    )


def enable_gradient_checkpointing(model: nn.Module, use_reentrant: bool) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    except TypeError:
        model.gradient_checkpointing_enable()


def _mark_norm_layers_fp32(model: nn.Module) -> None:
    norm_keywords = ("norm", "ln_f", "layer_norm", "rmsnorm")
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if any(keyword in class_name for keyword in norm_keywords):
            module.to(torch.float32)


def _require_input_grads(model: nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def prepare_model_for_manual_kbit_training(model: nn.Module, config: ExperimentConfig) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    _mark_norm_layers_fp32(model)
    _require_input_grads(model)
    if config.training.gradient_checkpointing:
        enable_gradient_checkpointing(model, config.training.gradient_checkpointing_use_reentrant)


def mark_quantized_model_as_adapter_trainable(model: nn.Module) -> None:
    # Transformers Trainer blocks fine-tuning of quantized base models unless it detects adapter-style training.
    # Manual LoRA is not a PEFT subclass, so we set the same internal marker explicitly.
    setattr(model, "_hf_peft_config_loaded", True)


def load_tokenizer(
    config: ExperimentConfig,
    *,
    tokenizer_path: str | Path | None = None,
    for_generation: bool = False,
):
    tokenizer_name = tokenizer_path or config.model.tokenizer_name or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=config.model.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left" if for_generation else "right"
    return tokenizer


def _build_model_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(config.model.torch_dtype),
        "low_cpu_mem_usage": config.model.low_cpu_mem_usage,
    }
    if config.model.attn_implementation:
        model_kwargs["attn_implementation"] = config.model.attn_implementation
    quantization_config = build_quantization_config(config)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        device_map = _resolve_device_map(config)
        if device_map is not None:
            model_kwargs["device_map"] = device_map
    return model_kwargs


def _load_peft_model(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    training: bool,
    adapter_path: str | Path | None,
) -> tuple[nn.Module, dict[str, Any]]:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

    if adapter_path is not None:
        peft_model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training)
        return peft_model, {"adapter_type": "peft_lora", "loaded_adapter": str(adapter_path)}

    if config.quantization.enabled:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
    elif config.training.gradient_checkpointing:
        enable_gradient_checkpointing(model, config.training.gradient_checkpointing_use_reentrant)
        _require_input_grads(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.adapter.rank,
        lora_alpha=config.adapter.alpha,
        lora_dropout=config.adapter.dropout,
        bias=config.adapter.bias,
        target_modules=config.adapter.target_modules,
        modules_to_save=config.adapter.modules_to_save or None,
    )
    peft_model = get_peft_model(model, peft_config)
    return peft_model, {"adapter_type": "peft_lora", "target_modules": config.adapter.target_modules}


def _load_manual_lora_model(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    training: bool,
    adapter_path: str | Path | None,
) -> tuple[nn.Module, dict[str, Any]]:
    if config.quantization.enabled:
        prepare_model_for_manual_kbit_training(model, config)
    elif training and config.training.gradient_checkpointing:
        enable_gradient_checkpointing(model, config.training.gradient_checkpointing_use_reentrant)
        _require_input_grads(model)

    replaced_modules = inject_manual_lora(
        model,
        target_modules=config.adapter.target_modules,
        rank=config.adapter.rank,
        alpha=config.adapter.alpha,
        dropout=config.adapter.dropout,
        bias=config.adapter.bias,
        freeze_base_model=config.adapter.freeze_base_model or config.quantization.enabled,
        modules_to_save=config.adapter.modules_to_save,
    )
    metadata: dict[str, Any] = {
        "adapter_type": "manual_lora",
        "target_modules": replaced_modules,
    }
    if adapter_path is not None:
        metadata["loaded_adapter"] = str(adapter_path)
        metadata["load_result"] = load_manual_lora_adapter(model, adapter_path)
    if config.quantization.enabled:
        mark_quantized_model_as_adapter_trainable(model)
    return model, metadata


def load_model_and_tokenizer(
    config: ExperimentConfig,
    *,
    training: bool,
    adapter_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    model_path_override: str | Path | None = None,
) -> tuple[Any, nn.Module, dict[str, Any]]:
    tokenizer = load_tokenizer(config, tokenizer_path=tokenizer_path, for_generation=not training)
    model_source = str(model_path_override or config.model.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_source, **_build_model_kwargs(config))
    model.config.use_cache = not training

    adapter_type = config.adapter.type
    adapter_info: dict[str, Any] = {"adapter_type": adapter_type}
    if adapter_type == "manual_lora":
        model, adapter_info = _load_manual_lora_model(model, config, training=training, adapter_path=adapter_path)
    elif adapter_type == "peft_lora":
        model, adapter_info = _load_peft_model(model, config, training=training, adapter_path=adapter_path)
    elif adapter_type == "none":
        if training and config.training.gradient_checkpointing:
            enable_gradient_checkpointing(model, config.training.gradient_checkpointing_use_reentrant)
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")

    adapter_info.update(count_parameters(model))
    return tokenizer, model, adapter_info


def resolve_saved_tokenizer_path(config: ExperimentConfig) -> Path | None:
    tokenizer_path = resolve_path(Path(config.training.output_dir) / "tokenizer")
    return tokenizer_path if tokenizer_path.exists() else None
