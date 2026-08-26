#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$SIM_DIR/data-causality-comparison/paper-evaluation00/corruption-budget}"
WORKERS="${WORKERS:-5}"

if (( WORKERS < 1 || WORKERS > 5 )); then
    echo "WORKERS must be between 1 and 5" >&2
    exit 2
fi

export PYTHONPATH="$SIM_DIR/.python-deps:$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-corruption-budget}"

cd "$REPO_DIR"
"$PYTHON_BIN" \
    simulations/src-gen_orderings/evaluate_corruption_budget_ablation.py \
    --ping_files \
        simulations/data-ping/aws-individual_pings.csv \
        simulations/data-ping/wp-hourly-2020-07-19-20.npz \
    --output_dir "$OUTPUT_DIR" \
    --deployments "${DEPLOYMENTS:-20}" \
    --trials_per_deployment "${TRIALS_PER_DEPLOYMENT:-100}" \
    --final_trials_per_deployment "${FINAL_TRIALS_PER_DEPLOYMENT:-20}" \
    --workers "$WORKERS" \
    --rho_ms "${RHO_MS:-0}" \
    --background_txs "${BACKGROUND_TXS:-300}" \
    --public_repetitions "${PUBLIC_REPETITIONS:-20}" \
    --bootstrap_repetitions "${BOOTSTRAP_REPETITIONS:-10000}" \
    --seed "${SEED:-762031}"

printf 'corruption-budget results: %s\n' "$OUTPUT_DIR"
