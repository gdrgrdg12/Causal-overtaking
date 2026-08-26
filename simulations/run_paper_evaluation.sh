#!/usr/bin/env bash
set -euo pipefail

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SIM_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULT_ROOT="${RESULT_ROOT:-$SIM_DIR/data-causality-comparison/paper-evaluation00}"
STAGES="${STAGES:-scaling,composition,budget,placement,rho,guard}"
DEPLOYMENTS="${DEPLOYMENTS:-20}"
TRIALS_PER_DEPLOYMENT="${TRIALS_PER_DEPLOYMENT:-20}"
PAIR_TRIALS_PER_DEPLOYMENT="${PAIR_TRIALS_PER_DEPLOYMENT:-100}"
WORKERS="${WORKERS:-5}"
PUBLIC_REPETITIONS="${PUBLIC_REPETITIONS:-20}"
BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-10000}"
SEED="${SEED:-762031}"
SCALING_DISSEMINATIONS="${SCALING_DISSEMINATIONS:-plain}"
SCALING_GOSSIP_P="${SCALING_GOSSIP_P:-0.1}"
SCALING_BATCH_SIZE="${SCALING_BATCH_SIZE:-5}"
PLACEMENT_RHO_MS="${PLACEMENT_RHO_MS:-0}"
PAPER_RHO_VALUES="${PAPER_RHO_VALUES:-1 2 5}"

if (( WORKERS < 1 || WORKERS > 5 )); then
    echo "WORKERS must be between 1 and 5" >&2
    exit 2
fi

if [[ -d "$SIM_DIR/.python-deps" ]]; then
    export PYTHONPATH="$SIM_DIR/.python-deps:$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
else
    export PYTHONPATH="$SIM_DIR/src-gen_orderings${PYTHONPATH:+:$PYTHONPATH}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-paper-evaluation}"

cd "$REPO_DIR"
mkdir -p "$RESULT_ROOT"

has_stage() {
    [[ ",$STAGES," == *",$1,"* ]]
}

dataset_file() {
    if [[ "$1" == "aws" ]]; then
        printf '%s\n' "$SIM_DIR/data-ping/aws-individual_pings.csv"
    else
        printf '%s\n' "$SIM_DIR/data-ping/wp-hourly-2020-07-19-20.npz"
    fi
}

dataset_layout() {
    if [[ "$1" == "aws" ]]; then
        printf '%s\n' "global-random"
    else
        printf '%s\n' "latency-clustered"
    fi
}

run_matched_shard() {
    local case_dir="$1" dataset="$2" nodes="$3" faults="$4"
    local layout="$5" placement="$6" rho="$7" deployment="$8"
    local ping_file shard
    ping_file="$(dataset_file "$dataset")"
    shard="$case_dir/shards/d$(printf '%02d' "$deployment")"
    if [[ -s "$shard/trial_results.csv" && -s "$shard/summary.csv" ]]; then
        echo "skip completed shard: $shard"
        return
    fi
    mkdir -p "$shard"
    "$PYTHON_BIN" simulations/src-gen_orderings/matched_multientry_evaluation.py \
        --ping_files "$ping_file" \
        --output_dir "$shard" \
        --deployments 1 \
        --deployment_start "$deployment" \
        --trials_per_deployment "$TRIALS_PER_DEPLOYMENT" \
        --num_nodes "$nodes" \
        --num_clients 15 \
        --replica_layout "$layout" \
        --replica_regions "$faults" \
        --observation_model corrupted-replica-direct \
        --corrupted_observers "$faults" \
        --corrupted_placement "$placement" \
        --allow_corrupted_region_colocation \
        --placement_victim_scope all-locations \
        --byzantine_reports "$faults" \
        --byzantine_report_mode pair-swap \
        --background_txs 300 \
        --entry_counts "$faults" \
        --training_snapshots 24 \
        --rho_ms "$rho" \
        --placement_rho_ms "$PLACEMENT_RHO_MS" \
        --selected_arm_mode full_union_only \
        --disseminations "$SCALING_DISSEMINATIONS" \
        --gossip_p "$SCALING_GOSSIP_P" \
        --batch_size "$SCALING_BATCH_SIZE" \
        --reporter_policies implementation \
        --public_repetitions "$PUBLIC_REPETITIONS" \
        --one_way_mode half \
        --clock_epsilon_ms 0 \
        --bootstrap_repetitions 100 \
        --conformance_cases 0 \
        --seed "$SEED"
}

run_matched_case() {
    local case_dir="$1" dataset="$2" nodes="$3" faults="$4"
    local layout="$5" placement="$6" rho="$7"
    local shards=()
    mkdir -p "$case_dir/shards"
    for ((deployment = 0; deployment < DEPLOYMENTS; deployment++)); do
        shards+=("$case_dir/shards/d$(printf '%02d' "$deployment")")
        run_matched_shard "$case_dir" "$dataset" "$nodes" "$faults" \
            "$layout" "$placement" "$rho" "$deployment" &
        while (( $(jobs -pr | wc -l) >= WORKERS )); do
            wait -n
        done
    done
    wait
    "$PYTHON_BIN" simulations/src-gen_orderings/merge_matched_multientry_results.py \
        --output_dir "$case_dir" \
        --shards "${shards[@]}" \
        --bootstrap_repetitions "$BOOTSTRAP_REPETITIONS" \
        --seed "$SEED"
}

if has_stage scaling; then
    for dataset in aws wp; do
        layout="$(dataset_layout "$dataset")"
        for spec in "9:2" "21:5" "101:25"; do
            nodes="${spec%%:*}"
            faults="${spec##*:}"
            run_matched_case \
                "$RESULT_ROOT/scaling/$dataset/n$nodes" \
                "$dataset" "$nodes" "$faults" "$layout" \
                history-optimized-global 0
        done
    done
fi

if has_stage composition; then
    for dataset in aws wp; do
        output="$RESULT_ROOT/composition/$dataset"
        ping_file="$(dataset_file "$dataset")"
        mkdir -p "$output"
        "$PYTHON_BIN" simulations/src-gen_orderings/evaluate_observer_composition.py \
            --ping_file "$ping_file" \
            --output_dir "$output" \
            --deployments "$DEPLOYMENTS" \
            --trials_per_deployment "$PAIR_TRIALS_PER_DEPLOYMENT" \
            --final_trials_per_deployment "$TRIALS_PER_DEPLOYMENT" \
            --workers "$WORKERS" \
            --replica_layout latency-clustered \
            --replica_regions 5 \
            --rho_ms 0 \
            --background_txs 300 \
            --public_repetitions "$PUBLIC_REPETITIONS" \
            --bootstrap_repetitions "$BOOTSTRAP_REPETITIONS" \
            --seed "$SEED"
    done
fi

if has_stage budget; then
    budget_output="$RESULT_ROOT/corruption-budget"
    if [[ -s "$budget_output/trial_results.csv.gz" \
       && -s "$budget_output/summary.csv" \
       && -s "$budget_output/composition_gain.csv" ]]; then
        echo "skip completed corruption-budget experiment: $budget_output"
    else
        OUTPUT_DIR="$budget_output" \
        WORKERS="$WORKERS" \
        DEPLOYMENTS="$DEPLOYMENTS" \
        TRIALS_PER_DEPLOYMENT="$PAIR_TRIALS_PER_DEPLOYMENT" \
        FINAL_TRIALS_PER_DEPLOYMENT="$TRIALS_PER_DEPLOYMENT" \
        PUBLIC_REPETITIONS="$PUBLIC_REPETITIONS" \
        BOOTSTRAP_REPETITIONS="$BOOTSTRAP_REPETITIONS" \
        SEED="$SEED" \
            bash simulations/run_corruption_budget_ablation.sh
    fi
fi

if has_stage placement; then
    for dataset in aws wp; do
        for placement in static-random region-medoid history-optimized-global; do
            run_matched_case \
                "$RESULT_ROOT/placement/$dataset/$placement" \
                "$dataset" 21 5 latency-clustered "$placement" 0
        done
    done
fi

if has_stage rho; then
    for dataset in aws wp; do
        layout="$(dataset_layout "$dataset")"
        for rho in $PAPER_RHO_VALUES; do
            run_matched_case \
                "$RESULT_ROOT/rho/$dataset/rho-$rho" \
                "$dataset" 21 5 "$layout" history-optimized-global "$rho"
        done
    done
fi

if has_stage guard; then
    for dataset in aws wp; do
        ping_file="$(dataset_file "$dataset")"
        layout="$(dataset_layout "$dataset")"
        for bound in same-hour previous-hour; do
            output="$RESULT_ROOT/guard-matched/$dataset/$bound"
            mkdir -p "$output"
            "$PYTHON_BIN" \
                simulations/src-gen_orderings/evaluate_corruption_guard_background.py \
                --ping_file "$ping_file" \
                --output_dir "$output" \
                --deployments "$DEPLOYMENTS" \
                --trials_per_deployment "$TRIALS_PER_DEPLOYMENT" \
                --background_txs 300 \
                --workers "$WORKERS" \
                --replica_layout "$layout" \
                --replica_regions 5 \
                --allow_corrupted_region_colocation \
                --execution_model section4-matched \
                --rho_ms 0 \
                --rho_floor_ms 0 \
                --bound_mode "$bound" \
                --profile_requirement 11 \
                --uncertainty_scale 1 \
                --public_repetitions "$PUBLIC_REPETITIONS" \
                --conformance_trials_per_deployment 1 \
                --bootstrap_repetitions "$BOOTSTRAP_REPETITIONS" \
                --seed "$SEED"
        done
    done

    conformance_build="/tmp/paper-evaluation-unspool"
    mkdir -p "$conformance_build"
    g++ -std=c++17 -O3 -Isimulations/include \
        simulations/src-unspooling_algo/unspool.cpp \
        -o "$conformance_build/unspool"
    for dataset in aws wp; do
        for bound in same-hour previous-hour; do
            output="$RESULT_ROOT/guard-matched/$dataset/$bound"
            "$PYTHON_BIN" \
                simulations/src-gen_orderings/run_corruption_guard_unspool_conformance.py \
                --binary "$conformance_build/unspool" \
                --result_dir "$output" \
                --work_dir "$conformance_build/$dataset-$bound" \
                --workers "$WORKERS"
        done
    done
fi

if [[ -f "$RESULT_ROOT/scaling/aws/n9/summary.csv" \
   && -f "$RESULT_ROOT/scaling/aws/n21/summary.csv" \
   && -f "$RESULT_ROOT/scaling/aws/n101/summary.csv" \
   && -f "$RESULT_ROOT/scaling/wp/n9/summary.csv" \
   && -f "$RESULT_ROOT/scaling/wp/n21/summary.csv" \
   && -f "$RESULT_ROOT/scaling/wp/n101/summary.csv" \
   && -f "$RESULT_ROOT/composition/aws/composition_summary.csv" \
   && -f "$RESULT_ROOT/composition/wp/composition_summary.csv" \
   && -f "$RESULT_ROOT/guard-matched/aws/previous-hour/component_summary.csv" \
   && -f "$RESULT_ROOT/guard-matched/wp/previous-hour/component_summary.csv" ]]; then
    "$PYTHON_BIN" simulations/src-gen_orderings/summarize_paper_evaluation.py \
        "$RESULT_ROOT"
    "$PYTHON_BIN" simulations/src-gen_orderings/plot_paper_evaluation.py \
        "$RESULT_ROOT"
    "$PYTHON_BIN" simulations/src-gen_orderings/validate_paper_evaluation.py \
        "$RESULT_ROOT" \
        --expected_deployments "$DEPLOYMENTS" \
        --expected_trials "$TRIALS_PER_DEPLOYMENT" \
        --expected_pair_trials "$PAIR_TRIALS_PER_DEPLOYMENT"
fi

printf 'paper evaluation results: %s\n' "$RESULT_ROOT"
