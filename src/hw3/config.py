from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hw3.prompts import DEFAULT_SYSTEM_PROMPT
from hw3.utils import ensure_parent_dir


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen2.5-7B"
    tokenizer_name: str | None = None
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    max_length: int = 768
    low_cpu_mem_usage: bool = True
    device_map: str | None = None


@dataclass
class QuantizationConfig:
    enabled: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_config_name: str = "main"
    train_split: str = "train"
    validation_file: str = "gsm8k_val.jsonl"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_proc: int = 1
    overwrite_cache: bool = False
    append_eos_token: bool = True
    use_chat_template: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    answer_prefix: str = "Answer:"


@dataclass
class AdapterConfig:
    enabled: bool = True
    type: str = "manual_lora"
    freeze_base_model: bool = True
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/default"
    overwrite_output_dir: bool = True
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    logging_strategy: str = "steps"
    save_strategy: str = "no"
    save_steps: int = 200
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    optim: str = "adamw_torch"
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    report_to: list[str] = field(default_factory=list)
    resume_from_checkpoint: str | None = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    num_beams: int = 1
    repetition_penalty: float = 1.0


@dataclass
class EvaluationConfig:
    batch_size: int = 1
    results_path: str | None = None
    metrics_path: str | None = None


@dataclass
class ExperimentConfig:
    experiment_name: str = "default"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(payload)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, raw_value = override.split("=", maxsplit=1)
        parsed_value = yaml.safe_load(raw_value)
        cursor = updated
        keys = key_path.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = parsed_value
    return updated


def _build_config(payload: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(
        experiment_name=payload.get("experiment_name", "default"),
        seed=payload.get("seed", 42),
        model=ModelConfig(**payload.get("model", {})),
        quantization=QuantizationConfig(**payload.get("quantization", {})),
        data=DataConfig(**payload.get("data", {})),
        adapter=AdapterConfig(**payload.get("adapter", {})),
        training=TrainingConfig(**payload.get("training", {})),
        generation=GenerationConfig(**payload.get("generation", {})),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
    )

    if not config.adapter.enabled:
        config.adapter.type = "none"

    if config.quantization.enabled and not (config.quantization.load_in_4bit or config.quantization.load_in_8bit):
        config.quantization.load_in_4bit = True

    if config.quantization.load_in_4bit and config.quantization.load_in_8bit:
        raise ValueError("Choose either 4-bit or 8-bit quantization, not both.")

    if config.quantization.enabled and config.adapter.type == "none":
        raise ValueError("Quantized training is only supported with LoRA adapters in this project.")

    if config.evaluation.results_path is None:
        config.evaluation.results_path = f"{config.training.output_dir}/results.jsonl"
    if config.evaluation.metrics_path is None:
        config.evaluation.metrics_path = f"{config.training.output_dir}/metrics.json"

    return config


def load_experiment_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if overrides:
        payload = _apply_overrides(payload, overrides)
    return _build_config(payload)


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    path_obj = ensure_parent_dir(path)
    with path_obj.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
