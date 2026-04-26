#!/usr/bin/env bash
set -euo pipefail

for d in /2022533131/CS190C-hw3-run /2022533131/CS190C-hw3-single /2022533131/CS190C-hw3-dual; do
  if [ -x "$d/.venv/bin/python" ]; then
    echo "===== fixing $d ====="
    "$d/.venv/bin/python" -m pip uninstall -y \
      torch torchvision torchaudio triton \
      nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
      nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
      nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
      nvidia-cusparselt-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 \
      nvidia-nvtx-cu12 >/dev/null 2>&1 || true
    "$d/.venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124
    "$d/.venv/bin/python" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
  fi
done
