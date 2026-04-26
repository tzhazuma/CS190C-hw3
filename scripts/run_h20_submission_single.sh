#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS=1 bash "${ROOT_DIR}/scripts/run_multi_experiments.sh" "${ROOT_DIR}/configs/suites/h20_single_recommended.yaml"
