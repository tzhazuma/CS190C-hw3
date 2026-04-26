# Implementation Plan

## Problem Breakdown

The assignment requires a training and evaluation pipeline for GSM8K with the following hard constraints:
- LoRA must be implemented manually.
- Training can only use the GSM8K training split.
- Evaluation must use the provided `gsm8k_val.jsonl`.
- The output must include `results.jsonl` with parsed numerical answers.
- The final validation accuracy target is at least `75%`.

## Chosen Engineering Approach

1. Build a unified YAML configuration system.
2. Implement manual LoRA injection for attention and MLP projections.
3. Keep a `peft`-based LoRA path as a comparison baseline.
4. Keep quantization as a switch so the same pipeline can run LoRA or QLoRA-style experiments.
5. Add multiple model configs so the same code can benchmark:
   - `Qwen/Qwen2.5-7B`
   - `Qwen/Qwen3.5-4B-Base`
6. Build a dedicated evaluation script that:
   - generates answers for each validation question
   - extracts the final number robustly
   - writes `results.jsonl`
   - computes accuracy
7. Add server-oriented shell scripts so the repository can be cloned and run directly on a GPU server.

## Why This Layout

- `src/hw3/manual_lora.py`: isolates the handwritten LoRA logic.
- `src/hw3/training.py`: keeps training pipeline changes independent of model choice.
- `src/hw3/evaluation.py`: keeps grading-format output deterministic and easy to rerun.
- `configs/experiments/*.yaml`: makes hyperparameter sweeps and model swaps explicit.
- `scripts/*.sh`: makes server usage reproducible without manual setup steps.

## Practical Tuning Strategy

Start with the faster configurations to validate the pipeline:
- `Qwen/Qwen3.5-4B-Base` + manual LoRA
- `Qwen/Qwen2.5-7B` + manual QLoRA

Then compare against the reference baselines:
- `Qwen/Qwen2.5-7B` + PEFT LoRA
- `Qwen/Qwen2.5-7B` + PEFT QLoRA

Likely knobs to improve accuracy if needed:
- increase `adapter.rank`
- increase epochs
- raise `max_length`
- expand target modules
- use the 7B model with 4-bit loading if GPU memory is limited
