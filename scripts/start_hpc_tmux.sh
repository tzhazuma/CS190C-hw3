#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_TARGET="${SSH_TARGET:-root@10.15.171.204}"
SSH_PORT="${SSH_PORT:-30911}"
REMOTE_DIR="${REMOTE_DIR:-/2022533131/CS190C-hw3}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/opt/conda/bin/python3.11}"
SESSION_NAME="${SESSION_NAME:-cs190c_hw3}"
MODE="${MODE:-single}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [ "${MODE}" = "dual" ]; then
  RUN_SCRIPT="bash scripts/run_h20_submission_dual.sh"
else
  RUN_SCRIPT="bash scripts/run_h20_submission_single.sh"
fi

REMOTE_CMD="cd ${REMOTE_DIR} && mkdir -p logs && export STUDENT_NAME=tangzhihao STUDENT_ID=2022533131 COURSE_NAME=CS190C PYTHON_BIN=${REMOTE_PYTHON} VENV_DIR=${REMOTE_DIR}/.venv TORCH_INSTALL_CMD=\"python -m pip install torch --index-url https://download.pytorch.org/whl/cu124\" MASTER_PORT=${MASTER_PORT} && bash scripts/setup_server.sh && ${RUN_SCRIPT} > logs/${SESSION_NAME}.log 2>&1"

ssh -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${SSH_TARGET}" "tmux kill-session -t ${SESSION_NAME} 2>/dev/null || true; tmux new-session -d -s ${SESSION_NAME} '${REMOTE_CMD}'; tmux ls; echo '[tmux] log: ${REMOTE_DIR}/logs/${SESSION_NAME}.log'"
