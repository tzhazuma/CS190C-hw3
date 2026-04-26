# CS190C Assignment 3: LoRA / QLoRA Fine-tuning on GSM8K

This repository implements the Assignment 3 pipeline for fine-tuning Qwen models on GSM8K.

It includes:
- a manual LoRA implementation from scratch for `nn.Linear`-style projections
- a reference `peft` LoRA version for comparison
- optional 4-bit QLoRA-style loading through `bitsandbytes`
- multi-model experiment configs, including `Qwen/Qwen2.5-7B` and `Qwen/Qwen3.5-4B-Base`
- training, evaluation, and server-friendly shell scripts
- a report template for the required write-up

## Assignment Mapping

The original assignment requires:
- manual LoRA implementation
- training only on the GSM8K training split
- evaluation on the provided `gsm8k_val.jsonl`
- robust answer parsing into `results.jsonl`
- reporting final accuracy and training hyperparameters in `README.md`

This repo satisfies those requirements while also preserving additional experiment options:
- `manual_lora`
- `manual_lora` + 4-bit quantization
- `peft_lora`
- `peft_lora` + 4-bit quantization
- multiple Qwen model configs

## Repository Structure

```text
configs/
  experiments/        # single experiment YAML files
  suites/             # multi-experiment suite files
docs/
  report_template.md  # report/write-up template
scripts/
  setup_server.sh     # create venv and install deps
  run_train_eval.sh   # train + evaluate one config
  run_multi_experiments.sh
  slurm_train_eval.sh
src/hw3/
  manual_lora.py      # hand-written LoRA injection
  training.py         # Trainer-based training pipeline
  evaluation.py       # gsm8k_val.jsonl evaluation pipeline
  parsing.py          # robust answer extraction
gsm8k_val.jsonl
```

## Environment Setup

Recommended on a Linux GPU server.

Python `3.10+` is recommended.

```bash
git clone https://github.com/tzhazuma/CS190C-hw3.git
cd CS190C-hw3
bash scripts/setup_server.sh
source .venv/bin/activate
```

If you want to control the Python executable or virtualenv path:

```bash
PYTHON_BIN=python3.10 VENV_DIR=$PWD/.venv bash scripts/setup_server.sh
```

If your server needs a CUDA-specific PyTorch installation, provide it explicitly:

```bash
TORCH_INSTALL_CMD='python -m pip install torch --index-url https://download.pytorch.org/whl/cu124' \
  bash scripts/setup_server.sh
```

## Single Experiment Run

Default manual LoRA run on `Qwen/Qwen2.5-7B`:

```bash
bash scripts/run_train_eval.sh configs/experiments/qwen25_7b_manual_lora.yaml
```

Direct Python usage:

```bash
export PYTHONPATH=$PWD/src
python scripts/train.py --config configs/experiments/qwen25_7b_manual_lora.yaml
python scripts/evaluate.py --config configs/experiments/qwen25_7b_manual_lora.yaml
```

Override any config field from the command line:

```bash
python scripts/train.py \
  --config configs/experiments/qwen25_7b_manual_lora.yaml \
  --override training.learning_rate=1e-4 \
  --override adapter.rank=32 \
  --override quantization.enabled=true \
  --override quantization.load_in_4bit=true
```

## Multi-Experiment Run

Run the provided suite:

```bash
bash scripts/run_multi_experiments.sh configs/suites/multi_model_experiments.yaml
```

Current suite includes:
- `Qwen/Qwen2.5-7B` + manual LoRA
- `Qwen/Qwen2.5-7B` + manual QLoRA
- `Qwen/Qwen2.5-7B` + PEFT LoRA
- `Qwen/Qwen2.5-7B` + PEFT QLoRA
- `Qwen/Qwen3.5-4B-Base` + manual LoRA
- `Qwen/Qwen3.5-4B-Base` + manual QLoRA
- `Qwen/Qwen3.5-4B-Base` + PEFT LoRA
- `Qwen/Qwen3.5-4B-Base` + PEFT QLoRA

## Important Configurations

### 1. Manual LoRA

`configs/experiments/qwen25_7b_manual_lora.yaml`

Key options:
- `adapter.type: manual_lora`
- `adapter.rank`
- `adapter.alpha`
- `adapter.dropout`
- `adapter.target_modules`

### 2. Manual QLoRA-style Run

`configs/experiments/qwen25_7b_manual_qlora.yaml`

Key options:
- `quantization.enabled: true`
- `quantization.load_in_4bit: true`
- `training.optim: paged_adamw_8bit`

### 3. PEFT LoRA Reference

`configs/experiments/qwen25_7b_peft_lora.yaml`

Key options:
- `adapter.type: peft_lora`

## Manual LoRA Implementation Notes

The hand-written implementation is in `src/hw3/manual_lora.py`.

Design summary:
- wraps target linear layers with `LoRALinear`
- keeps the original base projection as `base_layer`
- adds trainable low-rank matrices `lora_A` and `lora_B`
- computes `base_layer(x) + scale * B(A(dropout(x)))`
- freezes the base model and leaves only LoRA weights trainable by default

Supported targets are configured with module suffixes such as:
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

## Evaluation and Output Files

Evaluation reads the provided `gsm8k_val.jsonl` and writes:
- `results.jsonl`
- `metrics.json`

Each line in `results.jsonl` contains:

```json
{
  "question": "string",
  "ground_truth": "string",
  "model_output": "string",
  "parsed_answer": "string or null",
  "is_correct": true
}
```

Answer parsing is implemented in `src/hw3/parsing.py` and supports:
- standard `#### answer`
- fallback extraction from the last numeric-looking span
- normalization of commas, decimals, and simple fractions

## Server Usage

### Basic server workflow

```bash
git clone https://github.com/tzhazuma/CS190C-hw3.git
cd CS190C-hw3
bash scripts/setup_server.sh
bash scripts/run_train_eval.sh configs/experiments/qwen25_7b_manual_lora.yaml
```

### SLURM example

```bash
sbatch scripts/slurm_train_eval.sh configs/experiments/qwen25_7b_manual_lora.yaml
```

## Suggested Training Notes

You asked to preserve multiple options rather than locking one path. The practical tuning knobs are:
- switch between `manual_lora` and `peft_lora`
- toggle `quantization.enabled`
- increase `adapter.rank`
- increase `generation.max_new_tokens`
- switch `model.name_or_path`
- adjust batch size and accumulation based on GPU memory

For a 24GB to 48GB GPU, the most practical starting points are:
- `Qwen/Qwen2.5-7B` with manual QLoRA
- `Qwen/Qwen2.5-7B` with PEFT QLoRA
- `Qwen/Qwen3.5-4B-Base` with manual LoRA for faster iteration

## Final Results Section

Fill this section after running your best experiment.

| Item | Value |
| --- | --- |
| Best config | TODO |
| Final validation accuracy | TODO |
| Base model | TODO |
| Adapter type | TODO |
| Quantization | TODO |
| LoRA rank | TODO |
| LoRA alpha | TODO |
| Epochs | TODO |
| Batch size | TODO |
| Gradient accumulation | TODO |
| Learning rate | TODO |
| Hardware used | TODO |

## Deliverables Checklist

- [x] code for training and evaluation
- [x] manual LoRA implementation from scratch
- [x] PEFT LoRA reference implementation
- [x] quantization / QLoRA options
- [x] server scripts
- [x] report template
- [ ] final best `results.jsonl`
- [ ] final measured accuracy

## Report Template

Use `docs/report_template.md` or `reports/final_report_template.md` for your final write-up.
