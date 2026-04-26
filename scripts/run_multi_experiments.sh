#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE_PATH="${1:-${ROOT_DIR}/configs/suites/multi_model_experiments.yaml}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${SUITE_PATH}" != /* ]]; then
  SUITE_PATH="${ROOT_DIR}/${SUITE_PATH#./}"
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

python "${ROOT_DIR}/scripts/run_suite.py" \
  --suite "${SUITE_PATH}" \
  --num-gpus "${NUM_GPUS}" \
  --master-port-base "${MASTER_PORT}"

python "${ROOT_DIR}/scripts/summarize_results.py" --suite "${SUITE_PATH}"
