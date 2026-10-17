#!/usr/bin/env bash
set -euo pipefail
LINGBO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$LINGBO_ROOT"
exec "${LINGBO_PYTHON:-/tmp/lingbo-baseline-venv/bin/python}" -u -m evaluation.run "$@"
