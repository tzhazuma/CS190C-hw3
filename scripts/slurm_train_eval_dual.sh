#!/usr/bin/env bash
#SBATCH --job-name=cs190c-hw3-2gpu
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/experiments/qwen25_7b_manual_lora_h20_dual.yaml}"

NUM_GPUS=2 MASTER_PORT="${MASTER_PORT:-29501}" bash "${ROOT_DIR}/scripts/run_train_eval.sh" "${CONFIG_PATH}"
