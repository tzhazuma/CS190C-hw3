# CS190C Assignment 3 Submission README

Generated at: `{{generated_at}}`

## Student Info
- Student name: {{student_name}}
- Student ID: {{student_id}}
- Course: {{course_name}}

## Best Run
- Experiment: `{{best_experiment_name}}`
- Config: `{{best_config_path}}`
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

## Hardware And Software
- GPU: {{gpu_summary}}
- CUDA: {{cuda_version}}
- Python: {{python_version}}
- PyTorch: {{torch_version}}
- Transformers: {{transformers_version}}
- PEFT: {{peft_version}}

## Included Artifacts
- `results.jsonl` copied from `{{results_source_path}}`
- `final_report.md`
- `experiment_summary.md`
- source code snapshot (`src/`, `scripts/`, `configs/`, `docs/`)

## Experiment Summary
{{experiments_table}}

## Reproduction
```bash
bash scripts/setup_server.sh
{{best_run_command}}
```
