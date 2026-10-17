#!/usr/bin/env bash
set -euo pipefail

LINGBO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROBOTWIN_ROOT="$LINGBO_ROOT/RoboTwin"
export ROBOTWIN_ASSETS="${ROBOTWIN_ASSETS:-/srv/common_dataset/RoboTwin2.0/assets}"
export PYTHONPATH="$ROBOTWIN_ROOT/envs/curobo/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_HOME=/home/chenguanxu/.local/cuda-12.1
export PATH=/home/chenguanxu/miniconda3/envs/robotwin/bin:"$CUDA_HOME/bin":$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib":${LD_LIBRARY_PATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export SAPIEN_RENDER_DEVICE=cuda:0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TORCH_CUDA_ARCH_LIST=8.9

if [[ -n "${ROBOTWIN_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$ROBOTWIN_GPU"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | head -n1 | cut -d, -f1 | tr -d ' ')"
fi

python "$LINGBO_ROOT/tools/prepare_robotwin_assets.py"
cd "$ROBOTWIN_ROOT"
if [[ $# -eq 0 ]]; then
    exec python "$LINGBO_ROOT/tools/robotwin_smoke.py"
fi
exec "$@"
