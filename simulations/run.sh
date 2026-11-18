#!/usr/bin/env bash
# All dependencies and generated outputs stay inside the artifact.
set -euo pipefail
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"
source "$SIM_DIR/runtime.sh"
cd "$REPO_DIR"
export PYTHONPATH="$SIM_DIR/experiments:$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
STAGE="${1:-help}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_DIR/runs}"
DEPLOYMENTS="${DEPLOYMENTS:-20}"
TRIALS="${TRIALS:-20}"
PAIR_TRIALS="${PAIR_TRIALS:-100}"
WORKERS="${WORKERS:-4}"
BOOTSTRAP="${BOOTSTRAP:-10000}"
SIZES="${SIZES:-9 21 101}"
DATASETS="${DATASETS:-aws wp}"
read -r -a datasets <<< "$DATASETS"
read -r -a sizes <<< "$SIZES"
common=(--deployments "$DEPLOYMENTS" --trials "$TRIALS" --workers "$WORKERS")
mkdir -p "$RESULT_ROOT"

case "$STAGE" in
    attack|reporters|asymmetry)
        stages=(baseline reporters asymmetry)
        if [[ "$STAGE" == reporters ]]; then stages=(reporters); fi
        if [[ "$STAGE" == asymmetry ]]; then stages=(asymmetry); fi
        "$PYTHON_BIN" simulations/experiments/attack.py --out "$RESULT_ROOT/attack" \
            "${common[@]}" --datasets "${datasets[@]}" --sizes "${sizes[@]}" --stages "${stages[@]}" --bootstrap-repeats "$BOOTSTRAP"
        ;;
    capability)
        paths=()
        for ds in "${datasets[@]}"; do
            if [[ "$ds" == aws ]]; then paths+=("$SIM_DIR/data-ping/aws-individual_pings.csv")
            elif [[ "$ds" == wp ]]; then paths+=("$SIM_DIR/data-ping/wp-hourly-2020-07-19-20.npz")
            else echo "Unknown dataset: $ds" >&2; exit 2; fi
        done
        "$PYTHON_BIN" simulations/src-gen_orderings/evaluate_corruption_budget_ablation.py \
            --ping_files "${paths[@]}" --output_dir "$RESULT_ROOT/capability" \
            --deployments "$DEPLOYMENTS" --trials_per_deployment "$PAIR_TRIALS" \
            --final_trials_per_deployment "$TRIALS" --workers "$WORKERS" \
            --background_txs 300 --public_repetitions 20 --bootstrap_repetitions "$BOOTSTRAP" \
            --conformance_trials_per_deployment 1 --seed 762031
        ;;
    composition|placement|rho)
        for ds in "${datasets[@]}"; do
            if [[ "$ds" == aws ]]; then input="$SIM_DIR/data-ping/aws-individual_pings.csv"; layout=global-random
            elif [[ "$ds" == wp ]]; then input="$SIM_DIR/data-ping/wp-hourly-2020-07-19-20.npz"; layout=latency-clustered
            else echo "Unknown dataset: $ds" >&2; exit 2; fi
            if [[ "$STAGE" == composition ]]; then
                "$PYTHON_BIN" simulations/src-gen_orderings/evaluate_observer_composition.py \
                    --ping_file "$input" --output_dir "$RESULT_ROOT/composition/$ds" \
                    --deployments "$DEPLOYMENTS" --trials_per_deployment "$PAIR_TRIALS" \
                    --final_trials_per_deployment "$TRIALS" --workers "$WORKERS" \
                    --replica_layout latency-clustered --replica_regions 5 --rho_ms 0 \
                    --background_txs 300 --public_repetitions 20 --bootstrap_repetitions "$BOOTSTRAP" \
                    --conformance_trials_per_deployment 1 --seed 762031
                continue
            fi
            values=(0 1 2 5)
            if [[ "$STAGE" == placement ]]; then values=(static-random region-medoid history-optimized-global); layout=latency-clustered; fi
            for value in "${values[@]}"; do
                rho=0; placement=history-optimized-global
                if [[ "$STAGE" == rho ]]; then rho="$value"; else placement="$value"; fi
                "$PYTHON_BIN" simulations/src-gen_orderings/matched_multientry_evaluation.py \
                    --ping_files "$input" --output_dir "$RESULT_ROOT/$STAGE/$ds/$value" \
                    --deployments "$DEPLOYMENTS" --trials_per_deployment "$TRIALS" \
                    --num_nodes 21 --num_clients 15 --replica_layout "$layout" --replica_regions 5 \
                    --observation_model corrupted-replica-direct --corrupted_observers 5 \
                    --corrupted_placement "$placement" --allow_corrupted_region_colocation \
                    --placement_victim_scope all-locations --byzantine_reports 5 --byzantine_report_mode pair-swap \
                    --background_txs 300 --entry_counts 5 --training_snapshots 24 \
                    --rho_ms "$rho" --placement_rho_ms 0 --selected_arm_mode full_union_only \
                    --disseminations plain --reporter_policies implementation --public_repetitions 20 \
                    --one_way_mode half --clock_epsilon_ms 0 --bootstrap_repetitions "$BOOTSTRAP" \
                    --conformance_cases 1 --seed 762031
            done
        done
        ;;
    entries)
        "$PYTHON_BIN" simulations/experiments/entries.py --out "$RESULT_ROOT/entries" \
            "${common[@]}" --datasets "${datasets[@]}" --bootstrap-repeats "$BOOTSTRAP"
        ;;
    owd)
        "$PYTHON_BIN" simulations/experiments/owd.py --out "$RESULT_ROOT/owd" "${common[@]}"
        ;;
    gossip)
        "$PYTHON_BIN" simulations/experiments/gossip.py --out "$RESULT_ROOT/gossip" \
            "${common[@]}" --datasets "${datasets[@]}"
        ;;
    regional)
        "$PYTHON_BIN" simulations/experiments/regional.py --out "$RESULT_ROOT/regional" \
            "${common[@]}" --datasets "${datasets[@]}"
        ;;
    adaptive)
        "$PYTHON_BIN" simulations/experiments/adaptive.py --out "$RESULT_ROOT/adaptive" \
            "${common[@]}" --datasets "${datasets[@]}" --bootstrap-repeats "$BOOTSTRAP" \
            --train-cases "${TRAIN_CASES:-8}" --validation-cases "${VALIDATION_CASES:-3}"
        ;;
    benchmark)
        RESULT_DIR="$RESULT_ROOT/benchmark" bash simulations/rho-benchmark/run_rho_benchmark.sh
        ;;
    summarize)
        "$PYTHON_BIN" simulations/experiments/summarize.py "$RESULT_ROOT" --plots
        ;;
    smoke)
        export DEPLOYMENTS=1 TRIALS=1 PAIR_TRIALS=1 WORKERS=1 BOOTSTRAP=20 SIZES=21
        export RESULT_ROOT="${RESULT_ROOT}/smoke"
        for stage in attack entries owd gossip regional; do bash "$SIM_DIR/run.sh" "$stage"; done
        # Verify adaptive search with its full paper-sized 8/3 historical split.
        bash "$SIM_DIR/run.sh" adaptive
        "$PYTHON_BIN" simulations/experiments/check_unspool.py "$RESULT_ROOT"
        bash "$SIM_DIR/run.sh" summarize
        ;;
    all)
        for stage in attack capability composition placement rho entries owd gossip regional adaptive; do bash "$SIM_DIR/run.sh" "$stage"; done
        bash "$SIM_DIR/run.sh" summarize
        ;;
    help|--help|-h)
        echo 'Usage: bash simulations/run.sh {smoke|all|attack|capability|composition|placement|rho|entries|owd|gossip|regional|adaptive|reporters|asymmetry|benchmark|summarize}'
        echo 'Defaults: DEPLOYMENTS=20 TRIALS=20 PAIR_TRIALS=100 WORKERS=4 BOOTSTRAP=10000'
        echo 'Outputs: runs/ (override RESULT_ROOT).'
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
