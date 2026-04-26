#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/experiments/qwen25_7b_manual_lora.yaml}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

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
mkdir -p "${HF_HOME}" "${ROOT_DIR}/logs"

echo "[run] training config=${CONFIG_PATH}"
python "${ROOT_DIR}/scripts/train.py" --config "${CONFIG_PATH}"

echo "[run] evaluating config=${CONFIG_PATH}"
python "${ROOT_DIR}/scripts/evaluate.py" --config "${CONFIG_PATH}"
