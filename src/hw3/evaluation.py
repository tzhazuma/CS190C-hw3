from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from hw3.config import ExperimentConfig
from hw3.data import load_validation_dataset
from hw3.modeling import load_model_and_tokenizer, resolve_saved_tokenizer_path
from hw3.parsing import answers_match, extract_final_answer
from hw3.prompts import render_prompt
from hw3.utils import ensure_parent_dir, get_default_device, resolve_path, write_json, write_jsonl


def _batched_examples(examples: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def resolve_adapter_artifact(config: ExperimentConfig, adapter_path: str | Path | None = None) -> tuple[str | Path | None, str | Path | None]:
    output_dir = resolve_path(config.training.output_dir)
    tokenizer_path = resolve_saved_tokenizer_path(config)
    if adapter_path is not None:
        return adapter_path, tokenizer_path
    if config.adapter.type == "manual_lora":
        return output_dir / "adapter" / "manual_lora_adapter.pt", tokenizer_path
    if config.adapter.type == "peft_lora":
        return output_dir / "adapter", tokenizer_path
    model_dir = output_dir / "model"
    if model_dir.exists():
        return model_dir, tokenizer_path
    return None, tokenizer_path


def evaluate_experiment(config: ExperimentConfig, adapter_path: str | Path | None = None) -> dict[str, Any]:
    eval_dataset = load_validation_dataset(config)
    rows = [dict(example) for example in eval_dataset]
    resolved_adapter_path, tokenizer_path = resolve_adapter_artifact(config, adapter_path=adapter_path)

    model_path_override = None
    adapter_for_loader: str | Path | None = resolved_adapter_path
    if config.adapter.type == "none" and resolved_adapter_path is not None:
        model_path_override = resolved_adapter_path
        adapter_for_loader = None

    tokenizer, model, adapter_info = load_model_and_tokenizer(
        config,
        training=False,
        adapter_path=adapter_for_loader,
        tokenizer_path=tokenizer_path,
        model_path_override=model_path_override,
    )

    model.eval()
    device = get_default_device(model)
    results: list[dict[str, Any]] = []
    correct = 0
    batch_size = max(1, config.evaluation.batch_size)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.generation.max_new_tokens,
        "do_sample": config.generation.do_sample,
        "num_beams": config.generation.num_beams,
        "repetition_penalty": config.generation.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.generation.do_sample:
        generation_kwargs["temperature"] = config.generation.temperature
        generation_kwargs["top_p"] = config.generation.top_p

    progress = tqdm(_batched_examples(rows, batch_size), total=(len(rows) + batch_size - 1) // batch_size, desc="Evaluating")
    with torch.inference_mode():
        for batch in progress:
            prompts = [
                render_prompt(
                    tokenizer,
                    question=example["question"],
                    system_prompt=config.data.system_prompt,
                    answer_prefix=config.data.answer_prefix,
                    use_chat_template=config.data.use_chat_template,
                    answer=None,
                )
                for example in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.model.max_length,
                add_special_tokens=not bool(config.data.use_chat_template and getattr(tokenizer, "chat_template", None)),
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(**encoded, **generation_kwargs)
            prompt_length = encoded["input_ids"].shape[1]
            generated_only = generated[:, prompt_length:]
            decoded_outputs = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            for example, model_output in zip(batch, decoded_outputs, strict=True):
                parsed_prediction = extract_final_answer(model_output)
                ground_truth = extract_final_answer(example["answer"])
                is_correct = answers_match(parsed_prediction, ground_truth)
                correct += int(is_correct)
                results.append(
                    {
                        "question": example["question"],
                        "ground_truth": ground_truth,
                        "model_output": model_output.strip(),
                        "parsed_answer": parsed_prediction,
                        "is_correct": is_correct,
                    }
                )

    accuracy = correct / len(results) if results else 0.0
    results_path = resolve_path(config.evaluation.results_path)
    metrics_path = resolve_path(config.evaluation.metrics_path)
    if results_path is None or metrics_path is None:
        raise ValueError("Evaluation output paths must be configured.")

    ensure_parent_dir(results_path)
    ensure_parent_dir(metrics_path)
    write_jsonl(results_path, results)

    metrics = {
        "experiment_name": config.experiment_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "adapter_type": config.adapter.type,
        "adapter_info": adapter_info,
        "results_path": str(results_path),
    }
    write_json(metrics_path, metrics)
    return metrics
