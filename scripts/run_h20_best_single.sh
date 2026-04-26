#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS=1 bash "${ROOT_DIR}/scripts/run_train_eval.sh" "${ROOT_DIR}/configs/experiments/qwen25_7b_manual_lora_h20_single.yaml"
