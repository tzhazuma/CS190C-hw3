#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/experiments/qwen25_7b_manual_lora_h20_single.yaml}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH#./}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[error] virtualenv not found at ${VENV_DIR}. Run scripts/setup_server.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "${HF_HOME}" "${ROOT_DIR}/logs"

echo "[run] training config=${CONFIG_PATH}"
if [ "${NUM_GPUS}" -gt 1 ]; then
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT_DIR}/scripts/train.py" --config "${CONFIG_PATH}"
else
  python "${ROOT_DIR}/scripts/train.py" --config "${CONFIG_PATH}"
fi

echo "[run] evaluating config=${CONFIG_PATH}"
python "${ROOT_DIR}/scripts/evaluate.py" --config "${CONFIG_PATH}"

echo "[run] summarizing config=${CONFIG_PATH}"
python "${ROOT_DIR}/scripts/summarize_results.py" --config "${CONFIG_PATH}"
