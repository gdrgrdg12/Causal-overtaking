#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

"$PYTHON_BOOTSTRAP" -c '
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise SystemExit("Pinned dependencies require Python 3.10–3.12. Set PYTHON_BOOTSTRAP to a compatible interpreter.")
'
if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
    "$PYTHON_BOOTSTRAP" -m venv "$REPO_DIR/.venv"
fi

# Install only inside this project; do not use the user site or global pip cache.
export PYTHONNOUSERSITE=1
"$REPO_DIR/.venv/bin/python" -m pip --isolated --require-virtualenv \
    --disable-pip-version-check --no-cache-dir install \
    -r "$REPO_DIR/requirements.txt"
"$REPO_DIR/.venv/bin/python" -m pip --isolated --require-virtualenv \
    --disable-pip-version-check --no-cache-dir check
printf 'Ready. Activate with: source "%s/.venv/bin/activate"\n' "$REPO_DIR"
