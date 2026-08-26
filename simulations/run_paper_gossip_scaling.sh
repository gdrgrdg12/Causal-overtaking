#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLAIN_ROOT="${PLAIN_ROOT:-$SIM_DIR/data-causality-comparison/paper-evaluation00}"
RESULT_ROOT="${RESULT_ROOT:-$SIM_DIR/data-causality-comparison/paper-gossip-scaling00}"
WORKERS="${WORKERS:-5}"
GOSSIP_P="${GOSSIP_P:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-5}"

if (( WORKERS < 1 || WORKERS > 5 )); then
    echo "WORKERS must be between 1 and 5" >&2
    exit 2
fi

RESULT_ROOT="$RESULT_ROOT" \
STAGES=scaling \
WORKERS="$WORKERS" \
SCALING_DISSEMINATIONS=delayed \
SCALING_GOSSIP_P="$GOSSIP_P" \
SCALING_BATCH_SIZE="$BATCH_SIZE" \
    bash "$SIM_DIR/run_paper_evaluation.sh"

cd "$REPO_DIR"
if [[ -d "$SIM_DIR/.python-deps" ]]; then
    export PYTHONPATH="$SIM_DIR/.python-deps:$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-paper-gossip-scaling}"

"$PYTHON_BIN" simulations/src-gen_orderings/summarize_paper_gossip_scaling.py \
    --plain_root "$PLAIN_ROOT" \
    --delayed_root "$RESULT_ROOT" \
    --output_dir "$RESULT_ROOT" \
    --gossip_p "$GOSSIP_P" \
    --batch_size "$BATCH_SIZE"

printf 'gossip scaling results: %s\n' "$RESULT_ROOT"
