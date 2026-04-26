from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence

from hw3.config import ExperimentConfig
from hw3.prompts import render_prompt, should_add_special_tokens
from hw3.utils import resolve_path


IGNORE_INDEX = -100


def load_training_dataset(config: ExperimentConfig) -> Dataset:
    dataset = load_dataset(
        config.data.dataset_name,
        name=config.data.dataset_config_name,
        split=config.data.train_split,
    )
    if config.data.max_train_samples is not None:
        dataset = dataset.select(range(min(config.data.max_train_samples, len(dataset))))
    return dataset


def load_validation_dataset(config: ExperimentConfig) -> Dataset:
    validation_path = resolve_path(config.data.validation_file)
    if validation_path is None:
        raise ValueError("validation_file must be configured.")
    dataset = load_dataset("json", data_files=str(validation_path), split="train")
    if config.data.max_eval_samples is not None:
        dataset = dataset.select(range(min(config.data.max_eval_samples, len(dataset))))
    return dataset


def tokenize_training_dataset(dataset: Dataset, tokenizer, config: ExperimentConfig) -> Dataset:
    add_special_tokens = should_add_special_tokens(tokenizer, config.data.use_chat_template)

    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        prompt_text = render_prompt(
            tokenizer,
            question=example["question"],
            system_prompt=config.data.system_prompt,
            answer_prefix=config.data.answer_prefix,
            use_chat_template=config.data.use_chat_template,
            answer=None,
        )
        full_text = render_prompt(
            tokenizer,
            question=example["question"],
            system_prompt=config.data.system_prompt,
            answer_prefix=config.data.answer_prefix,
            use_chat_template=config.data.use_chat_template,
            answer=example["answer"],
        )

        if config.data.append_eos_token and tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
            full_text = f"{full_text}{tokenizer.eos_token}"

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=config.model.max_length,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=config.model.max_length,
        )

        input_ids = full_tokens["input_ids"]
        prompt_length = min(len(prompt_tokens["input_ids"]), len(input_ids))
        labels = input_ids.copy()
        labels[:prompt_length] = [IGNORE_INDEX] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    return dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        num_proc=config.data.num_proc if config.data.num_proc > 1 else None,
        load_from_cache_file=not config.data.overwrite_cache,
        desc="Tokenizing GSM8K training split",
    )


@dataclass
class SupervisedDataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = [torch.tensor(feature["input_ids"], dtype=torch.long) for feature in features]
        attention_masks = [torch.tensor(feature["attention_mask"], dtype=torch.long) for feature in features]
        labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
            "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
        }
