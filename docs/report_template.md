# CS190C Assignment 3 Report Template

## Overview
- Student name:
- Student ID:
- Course:
- Date:

## Assignment Goal
- Base task: fine-tune Qwen on GSM8K with manually implemented LoRA.
- Validation file: `gsm8k_val.jsonl`
- Required metric: accuracy `>= 75%`

## Environment
- GPU:
- CUDA version:
- Python version:
- PyTorch version:
- Transformers version:
- PEFT version:

## Final Best Run
- Config file:
- Base model:
- Adapter type: `manual_lora` / `peft_lora`
- Quantization: on / off
- Validation accuracy:
- Output directory:

## Hyperparameters
| Item | Value |
| --- | --- |
| rank (r) |  |
| alpha |  |
| dropout |  |
| learning rate |  |
| epochs |  |
| per-device batch size |  |
| gradient accumulation |  |
| max length |  |

## Experiments
| Experiment | Model | Adapter | Quantization | Accuracy | Notes |
| --- | --- | --- | --- | --- | --- |
| baseline-1 |  |  |  |  |  |
| baseline-2 |  |  |  |  |  |
| final |  |  |  |  |  |

## Manual LoRA Implementation Notes
- Which modules were wrapped:
- How `A` and `B` matrices were initialized:
- How trainable parameters were restricted:
- Any stability tricks used:

## Answer Parsing Strategy
- Primary extraction rule:
- Fallback extraction rule:
- Handling commas / decimals / malformed outputs:

## Reproduction Steps
```bash
bash scripts/setup_server.sh
bash scripts/run_train_eval.sh configs/experiments/qwen25_7b_manual_lora.yaml
```

## Deliverables Checklist
- [ ] `README.md`
- [ ] `results.jsonl`
- [ ] training / evaluation code
- [ ] manual LoRA implementation
- [ ] report / notes
