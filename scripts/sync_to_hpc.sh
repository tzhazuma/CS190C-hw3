#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_TARGET="${SSH_TARGET:-root@10.15.171.204}"
SSH_PORT="${SSH_PORT:-30911}"
REMOTE_DIR="${REMOTE_DIR:-/2022533131/CS190C-hw3}"

echo "[sync] target=${SSH_TARGET}:${REMOTE_DIR}"
ssh -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${SSH_TARGET}" "python3 - <<'PY'
import os
import shutil

path = os.path.abspath('${REMOTE_DIR}')
if os.path.exists(path):
    shutil.rmtree(path)
os.makedirs(path, exist_ok=True)
print(path)
PY"

COPYFILE_DISABLE=1 tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='outputs' \
  --exclude='logs' \
  --exclude='hf_cache' \
  --exclude='submission' \
  --exclude='*.zip' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  -C "${ROOT_DIR}" -czf - . | \
  ssh -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${SSH_TARGET}" "tar -xzf - -C \"${REMOTE_DIR}\""

ssh -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${SSH_TARGET}" "find \"${REMOTE_DIR}\" \( -name '._*' -o -name '.DS_Store' \) -delete && ls -la \"${REMOTE_DIR}\" | sed -n '1,40p'"
