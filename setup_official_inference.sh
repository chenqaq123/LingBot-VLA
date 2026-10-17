#!/usr/bin/env bash
set -euo pipefail
LINGBO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LINGBO_PY="${LINGBO_PYTHON:-/tmp/lingbo-baseline-venv/bin/python}"
LINGBO_SIM_PY="${LINGBO_SIM_PYTHON:-/home/chenguanxu/miniconda3/envs/robotwin/bin/python}"
# Rebuild the isolated environment if its temporary directory was cleared.
if [[ ! -x "$LINGBO_PY" ]]; then
    LINGBO_BASE_PY="${LINGBO_BASE_PYTHON:-/home/chenguanxu/miniconda3/envs/vla-smolvla/bin/python}"
    "$LINGBO_BASE_PY" -m venv --system-site-packages "$(dirname -- "$(dirname -- "$LINGBO_PY")")"
fi
"$LINGBO_PY" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3,11), 'This prebuilt wheel needs Python 3.11'
assert torch.__version__.split('+')[0].startswith('2.7.'), 'This wheel needs PyTorch 2.7'
assert torch._C._GLIBCXX_USE_CXX11_ABI, 'This wheel needs the CXX11 ABI'
PY
"$LINGBO_PY" -m pip install -r "$LINGBO_ROOT/requirements-inference.txt"
"$LINGBO_PY" -m pip install --no-deps msgpack==1.1.1 websockets==15.0.1 \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
"$LINGBO_SIM_PY" -m pip install --no-deps --upgrade --target /tmp/lingbo-official-sim-deps \
  msgpack==1.1.1 websockets==15.0.1
