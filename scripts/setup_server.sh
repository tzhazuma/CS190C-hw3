#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
HF_CACHE_DIR="${HF_HOME:-${ROOT_DIR}/hf_cache}"

echo "[setup] root=${ROOT_DIR}"
echo "[setup] python=${PYTHON_BIN}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

if ! "${VENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  if "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  elif "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
    echo "[setup] venv missing pip; falling back to virtualenv bootstrap"
    rm -rf "${VENV_DIR}"
    "${PYTHON_BIN}" -m pip install --upgrade virtualenv
    "${PYTHON_BIN}" -m virtualenv -p "${PYTHON_BIN}" "${VENV_DIR}"
  else
    echo "[setup] unable to provision pip inside virtual environment" >&2
    echo "[setup] install pip or virtualenv for ${PYTHON_BIN} and retry" >&2
    exit 1
  fi
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if ! python -c "import torch" >/dev/null 2>&1; then
  if [ -n "${TORCH_INSTALL_CMD:-}" ]; then
    echo "[setup] installing torch via TORCH_INSTALL_CMD"
    eval "${TORCH_INSTALL_CMD}"
  else
    echo "[setup] torch not found; installing default pip torch wheel"
    echo "[setup] if your server needs a CUDA-specific wheel, rerun with TORCH_INSTALL_CMD set"
    python -m pip install torch
  fi
fi

HF_HOME="${HF_CACHE_DIR}" python -m pip install -r "${ROOT_DIR}/requirements.txt"

mkdir -p "${HF_CACHE_DIR}" "${ROOT_DIR}/outputs" "${ROOT_DIR}/logs"

echo "[setup] completed"
echo "[setup] activate with: source ${VENV_DIR}/bin/activate"
