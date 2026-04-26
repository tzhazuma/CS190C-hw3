# H20 Operations Guide

This document gives a practical runbook for your H20 server.

## Recommended Default Choice

If the goal is the best balance of accuracy and simplicity on H20, start with:

- single GPU: `configs/experiments/qwen25_7b_manual_lora_h20_single.yaml`
- two GPUs: `configs/experiments/qwen25_7b_manual_lora_h20_dual.yaml`

Why this is the default recommendation:
- it exactly satisfies the assignment requirement of a manual LoRA implementation
- H20 has enough memory to run Qwen2.5-7B without 4-bit quantization
- avoiding quantization removes one failure surface during training and evaluation
- the config keeps a longer context window and larger batch than the generic baseline

## Alternative Strong Baseline

If you want faster experimentation or higher throughput:

- single GPU QLoRA: `configs/experiments/qwen25_7b_manual_qlora_h20_single.yaml`
- dual GPU QLoRA: `configs/experiments/qwen25_7b_manual_qlora_h20_dual.yaml`

## One-Time Setup

```bash
git clone https://github.com/tzhazuma/CS190C-hw3.git
cd CS190C-hw3
bash scripts/setup_server.sh
source .venv/bin/activate
```

If CUDA-specific torch installation is needed:

```bash
TORCH_INSTALL_CMD='python -m pip install torch --index-url https://download.pytorch.org/whl/cu124' \
  bash scripts/setup_server.sh
```

## Single-GPU Run

```bash
bash scripts/run_h20_best_single.sh
```

Equivalent explicit command:

```bash
NUM_GPUS=1 bash scripts/run_train_eval.sh configs/experiments/qwen25_7b_manual_lora_h20_single.yaml
```

## Two-GPU Run

```bash
bash scripts/run_h20_best_dual.sh
```

Equivalent explicit command:

```bash
NUM_GPUS=2 MASTER_PORT=29501 bash scripts/run_train_eval.sh configs/experiments/qwen25_7b_manual_lora_h20_dual.yaml
```

## Recommended Sweep on H20

Single GPU:

```bash
NUM_GPUS=1 bash scripts/run_multi_experiments.sh configs/suites/h20_single_recommended.yaml
```

Two GPUs:

```bash
NUM_GPUS=2 MASTER_PORT=29501 bash scripts/run_multi_experiments.sh configs/suites/h20_dual_recommended.yaml
```

## One-Command Submission Workflow

Single GPU:

```bash
STUDENT_NAME="Your Name" \
STUDENT_ID="12345678" \
COURSE_NAME="CS190C" \
bash scripts/run_h20_submission_single.sh
```

Two GPUs:

```bash
STUDENT_NAME="Your Name" \
STUDENT_ID="12345678" \
COURSE_NAME="CS190C" \
MASTER_PORT=29501 \
bash scripts/run_h20_submission_dual.sh
```

These commands:
- run the recommended experiment suite
- summarize all completed experiments
- automatically select the best run by accuracy
- update the repository root `README.md` final results section
- generate autofilled `submission/README.md` and `submission/final_report.md`
- copy the best `results.jsonl`
- build `submission/CS190C-hw3-submission.zip`

If you already know which run is best and only want to refresh the root README:

```bash
python scripts/autofill_root_readme.py --config configs/experiments/qwen25_7b_manual_lora_h20_single.yaml
```

## Result Artifacts

Each experiment writes to its own `outputs/<experiment_name>/` directory.

Important files:
- `training_metrics.json`
- `metrics.json`
- `results.jsonl`
- `adapter/` or `model/`
- `config.resolved.yaml`

Suite runs also generate summary files under `reports/`:
- `*_summary.json`
- `*_summary.csv`
- `*_summary.md`

## Suggested Decision Rule

After the recommended sweep finishes:

1. Open the generated summary markdown in `reports/`.
2. Pick the highest-accuracy experiment.
3. Copy its config name, metrics, and hardware info into `README.md` and the report template.
4. Use that experiment's `results.jsonl` as the submission artifact.
