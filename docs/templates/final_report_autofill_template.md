# CS190C Assignment 3 Final Report

Generated at: `{{generated_at}}`

## Overview
- Student name: {{student_name}}
- Student ID: {{student_id}}
- Course: {{course_name}}

## Best Run
- Experiment: `{{best_experiment_name}}`
- Config file: `{{best_config_path}}`
- Base model: `{{base_model}}`
- Adapter type: `{{adapter_type}}`
- Quantization: `{{quantization}}`
- Validation accuracy: `{{accuracy_percent}}` (`{{correct}}/{{total}}`)
- Output directory: `{{output_dir}}`

## Hyperparameters
| Item | Value |
| --- | --- |
| rank (r) | {{rank}} |
| alpha | {{alpha}} |
| dropout | {{dropout}} |
| learning rate | {{learning_rate}} |
| epochs | {{epochs}} |
| per-device batch size | {{batch_size}} |
| gradient accumulation | {{gradient_accumulation}} |
| max length | {{max_length}} |

## Environment
- GPU: {{gpu_summary}}
- CUDA version: {{cuda_version}}
- Python version: {{python_version}}
- PyTorch version: {{torch_version}}
- Transformers version: {{transformers_version}}
- PEFT version: {{peft_version}}

## Automatic Selection Logic
- Best experiment is selected by highest validation accuracy from completed runs.
- Tie-break is deterministic by experiment name ordering after accuracy sorting.
- Selected results file: `{{results_source_path}}`

## Manual LoRA Implementation Notes
- Target modules: {{target_modules}}
- Low-rank update form: `base(x) + scale * B(A(dropout(x)))`
- Trainable parameters: LoRA matrices only by default; base model remains frozen unless explicitly configured.

## Answer Parsing Strategy
- Primary extraction rule: extract the last valid `#### <answer>` pattern.
- Fallback extraction rule: use the last numeric-looking span in the generated text.
- Normalization: strip commas, normalize decimals, support simple fractions.

## Experiment Summary
{{experiments_table}}

## Best Run Reproduction
```bash
bash scripts/setup_server.sh
{{best_run_command}}
```

## Observations
- The selected experiment was chosen automatically because it achieved the highest measured validation accuracy.
- For H20, non-quantized manual LoRA on Qwen2.5-7B is the default recommended starting point, while QLoRA variants are useful throughput baselines.
