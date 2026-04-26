from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from transformers import Trainer, TrainingArguments

from hw3.config import ExperimentConfig, config_to_dict, dump_experiment_config
from hw3.data import SupervisedDataCollator, load_training_dataset, tokenize_training_dataset
from hw3.manual_lora import save_manual_lora_adapter
from hw3.modeling import load_model_and_tokenizer
from hw3.utils import count_parameters, ensure_dir, resolve_path, set_global_seed, write_json


def _build_training_arguments(config: ExperimentConfig) -> TrainingArguments:
    save_strategy = config.training.save_strategy
    if config.adapter.type == "manual_lora" and save_strategy != "no":
        save_strategy = "no"

    argument_values = {
        "output_dir": config.training.output_dir,
        "overwrite_output_dir": config.training.overwrite_output_dir,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": config.training.max_steps,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "warmup_ratio": config.training.warmup_ratio,
        "lr_scheduler_type": config.training.lr_scheduler_type,
        "logging_steps": config.training.logging_steps,
        "logging_strategy": config.training.logging_strategy,
        "save_strategy": save_strategy,
        "save_steps": config.training.save_steps,
        "save_total_limit": config.training.save_total_limit,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": config.training.gradient_checkpointing_use_reentrant},
        "optim": config.training.optim,
        "max_grad_norm": config.training.max_grad_norm,
        "dataloader_num_workers": config.training.dataloader_num_workers,
        "remove_unused_columns": config.training.remove_unused_columns,
        "ddp_find_unused_parameters": config.training.ddp_find_unused_parameters,
        "report_to": config.training.report_to,
        "seed": config.seed,
        "save_safetensors": True,
    }
    supported_parameters = inspect.signature(TrainingArguments.__init__).parameters
    filtered_values = {key: value for key, value in argument_values.items() if key in supported_parameters}
    return TrainingArguments(**filtered_values)


def train_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    output_dir = ensure_dir(resolve_path(config.training.output_dir))

    tokenizer, model, adapter_info = load_model_and_tokenizer(config, training=True)
    raw_train_dataset = load_training_dataset(config)
    train_dataset = tokenize_training_dataset(raw_train_dataset, tokenizer, config)
    data_collator = SupervisedDataCollator(tokenizer)
    training_args = _build_training_arguments(config)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    trainer.save_state()

    tokenizer.save_pretrained(output_dir / "tokenizer")
    dump_experiment_config(config, output_dir / "config.resolved.yaml")

    if config.adapter.type == "manual_lora":
        save_manual_lora_adapter(
            model,
            output_dir / "adapter",
            metadata={
                "experiment_name": config.experiment_name,
                "base_model": config.model.name_or_path,
                "adapter": config_to_dict(config)["adapter"],
                "adapter_info": adapter_info,
            },
        )
    elif config.adapter.type == "peft_lora":
        model.save_pretrained(output_dir / "adapter")
    else:
        trainer.save_model(output_dir / "model")

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "experiment_name": config.experiment_name,
            "train_examples": len(raw_train_dataset),
            **count_parameters(model),
            "adapter_type": config.adapter.type,
        }
    )
    write_json(output_dir / "training_metrics.json", metrics)
    return metrics
