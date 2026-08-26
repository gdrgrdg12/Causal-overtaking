#!/usr/bin/env python3
"""Sweep actual corruptions, active observers, and forged reports at n=21."""

import argparse
import json
import math
import os
from multiprocessing import Pool
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate_corruption_aware_guard as guard
import evaluate_corruption_guard_background as background
import evaluate_observer_composition as composition
import matched_multientry_evaluation as matched


STRATEGIES = (
    ("observers_truthful", lambda c: (c, 0)),
    ("single_observer_reports", lambda c: (1, c)),
    ("observers_one_report", lambda c: (c, 1)),
    ("joint", lambda c: (c, c)),
)


def deployment_args(args, corruptions):
    return SimpleNamespace(
        seed=args.seed,
        num_nodes=21,
        num_clients=15,
        entry_counts=[corruptions],
        replica_layout=args.replica_layout,
        replica_regions=5,
        observation_model="corrupted-replica-direct",
        corrupted_observers=corruptions,
        corrupted_placement="history-optimized-global",
        allow_corrupted_region_colocation=True,
        placement_victim_scope="all-locations",
        relay_entry_count=args.location_pool_size,
        byzantine_reports=corruptions,
        rho_ms=args.rho_ms,
        training_snapshots=24,
    )


def pair_snapshot(trace, window, deployment, deployment_id, trial, args,
                  corruptions):
    if trace.is_hourly_snapshot:
        trace.set_snapshot_context(
            "execution", args.seed, "trial", deployment_id, trial
        )
        epoch = trace.active_snapshot["execution"]
        hour = window.hours[epoch]
    else:
        epoch = None
        hour = "odd-sample execution partition"
    victim = matched.stable_rng(
        args.seed, trace.dataset, deployment_id, trial, "victim"
    ).choice(deployment["clients"])
    replicas = deployment["replicas"]
    t_arrivals = matched.T_GEN_TIME + np.asarray([
        window.sample(
            epoch, victim, replica, args.seed,
            deployment_id, trial, "T", replica,
        )
        for replica in replicas
    ], dtype=np.float64)
    corrupted = matched.corrupted_replica_indices(
        deployment, deployment_args(args, corruptions)
    )
    singleton_arrivals = {}
    for corrupted_index in corrupted:
        source = replicas[corrupted_index]
        observed = t_arrivals[corrupted_index]
        singleton_arrivals[corrupted_index] = np.asarray([
            observed
            + args.rho_ms
            + window.sample(
                epoch, source, replica, args.seed,
                deployment_id, trial, "attack-F", source, replica,
            )
            for replica in replicas
        ], dtype=np.float64)
    return {
        "epoch": epoch,
        "hour": hour,
        "victim": victim,
        "t_arrivals": t_arrivals,
        "attack_f_arrivals": np.min(
            np.stack(list(singleton_arrivals.values())), axis=0
        ),
        "corrupted": corrupted,
        "singleton_arrivals": singleton_arrivals,
    }


def evaluate_pair(deployment, snapshot, f_arrivals, fake_reporters):
    reporters = np.asarray(deployment["reporter_order"][:16], dtype=int)
    fake_set = set(fake_reporters)
    fake_mask = np.asarray(
        [index in fake_set for index in reporters], dtype=bool
    )
    corrupted = set(snapshot["corrupted"])
    selected_t = snapshot["t_arrivals"][reporters]
    selected_f = guard.encode_t_before_f_ties(
        selected_t, np.asarray(f_arrivals, dtype=np.float64)[reporters]
    )
    behavior = "pair-swap" if fake_set else "honest"
    report_t, report_f = guard.apply_report_behavior(
        selected_t, selected_f, fake_mask, behavior
    )
    w_ft, w_tf, strict_edge = guard.strict_edge(report_t, report_f)
    honest_support = sum(
        index not in corrupted
        and selected_f[position] < selected_t[position] - guard.EPS_MS
        for position, index in enumerate(reporters)
    )
    return {
        "w_FT": w_ft,
        "w_TF": w_tf,
        "pair_tie": int(w_ft == w_tf),
        "strict_edge": strict_edge,
        "selected_honest_support": int(honest_support),
    }


def evaluate_strategy(deployment, snapshot, singletons, strategy, k, b,
                      deployment_id, trial, args, background_arrivals):
    corrupted = snapshot["corrupted"]
    active_observers = corrupted[:k]
    fake_reporters = corrupted[:b]
    f_arrivals = np.min(
        np.stack([singletons[index] for index in active_observers]), axis=0
    )
    pair = evaluate_pair(deployment, snapshot, f_arrivals, fake_reporters)
    row = {
        "strategy": strategy,
        "k_observers": k,
        "b_fake_reports": b,
        "selected_honest_support": pair["selected_honest_support"],
        "w_FT": pair["w_FT"],
        "w_TF": pair["w_TF"],
        "pair_tie": pair["pair_tie"],
        "strict_edge": pair["strict_edge"],
        "artifact_direct_edge_FT": math.nan,
        "artifact_direct_edge_TF": math.nan,
        "same_scc_rate": math.nan,
        "final_F_before_T_rate": math.nan,
        "strict_edge_and_final_rate": math.nan,
    }
    if background_arrivals is None:
        return row
    final_snapshot = dict(snapshot)
    final_snapshot["attack_f_arrivals"] = f_arrivals
    final = background.evaluate_attack_unspool(
        deployment, final_snapshot, background_arrivals,
        {"w_FT": pair["w_FT"], "w_TF": pair["w_TF"], "deferred_edge": 0},
        deployment_id, trial, args,
        "pair-swap" if b else "honest",
        byzantine_indices=fake_reporters,
    )
    if pair["strict_edge"] and not final["raw_direct_edge_FT"]:
        raise AssertionError("full graph dropped a strict F/T edge")
    row.update({
        "artifact_direct_edge_FT": final["raw_direct_edge_FT"],
        "artifact_direct_edge_TF": final["raw_direct_edge_TF"],
        "same_scc_rate": final["raw_same_scc_rate"],
        "final_F_before_T_rate": final["raw_final_F_before_T_rate"],
        "strict_edge_and_final_rate": (
            pair["strict_edge"] * final["raw_final_F_before_T_rate"]
        ),
    })
    return row


def run_case(task):
    ping_file, deployment_id, corruptions, args_dict = task
    args = SimpleNamespace(**args_dict)
    trace = matched.LatencyTrace(ping_file, "half", 0.15)
    args.location_pool_size = len(trace.locations)
    window = composition.load_window(trace, ping_file)
    dep_args = deployment_args(args, corruptions)
    deployment = matched.make_deployment(trace, deployment_id, dep_args)
    corrupted = deployment["corrupted_replica_indices"]
    if len(corrupted) != corruptions:
        raise AssertionError("wrong actual corruption count")
    if not set(corrupted).issubset(deployment["reporter_order"][:16]):
        raise AssertionError("actual corruptions left the selected report set")

    rows = []
    for trial in range(args.trials_per_deployment):
        snapshot = pair_snapshot(
            trace, window, deployment, deployment_id, trial, args, corruptions
        )
        background_arrivals = None
        if trial < args.final_trials_per_deployment:
            _, _, background_arrivals = background.generate_background(
                window, snapshot, deployment, deployment_id, trial,
                args.background_txs, args.seed,
            )
        for strategy, parameter_fn in STRATEGIES:
            k, b = parameter_fn(corruptions)
            if max(k, b) != corruptions:
                raise AssertionError(
                    f"{strategy} requires {max(k, b)} corruptions, not "
                    f"c={corruptions}"
                )
            result = evaluate_strategy(
                deployment, snapshot, snapshot["singleton_arrivals"],
                strategy, k, b, deployment_id, trial, args,
                background_arrivals,
            )
            rows.append({
                "dataset": trace.dataset,
                "deployment": deployment_id,
                "trial": trial,
                "protocol_n": 21,
                "protocol_f": 5,
                "actual_corruptions": corruptions,
                "required_corruptions": max(k, b),
                "victim": snapshot["victim"],
                "execution_snapshot_hour": snapshot["hour"],
                "reporter_ids": ";".join(
                    str(value) for value in deployment["reporter_order"][:16]
                ),
                "corrupted_ids": ";".join(str(value) for value in corrupted),
                "corrupted_locations": ";".join(
                    deployment["replicas"][index] for index in corrupted
                ),
                **result,
            })
        if corruptions == 1:
            report_rows = rows[-3:]
            comparable = (
                "w_FT", "w_TF", "strict_edge", "artifact_direct_edge_FT",
                "artifact_direct_edge_TF", "same_scc_rate",
                "final_F_before_T_rate", "strict_edge_and_final_rate",
            )
            for metric in comparable:
                values = np.asarray(
                    [row[metric] for row in report_rows], dtype=np.float64
                )
                finite = values[np.isfinite(values)]
                if len(finite) and not np.allclose(
                    finite, finite[0], rtol=0, atol=1e-12
                ):
                    raise AssertionError(
                        f"c=1 aliases disagree for {metric}: {values}"
                    )
    manifest = {
        "dataset": trace.dataset,
        "deployment": deployment_id,
        "actual_corruptions": corruptions,
        "replicas": json.dumps(deployment["replicas"]),
        "clients": json.dumps(deployment["clients"]),
        "reporter_order": json.dumps(deployment["reporter_order"]),
        "corrupted_ids": json.dumps(corrupted),
        "corrupted_locations": json.dumps([
            deployment["replicas"][index] for index in corrupted
        ]),
        "placement_info": json.dumps(deployment["corrupted_placement_info"]),
    }
    return rows, manifest


def bootstrap(frame, metric, args, offset):
    value = float(frame[metric].mean())
    low, high = matched.hierarchical_bootstrap(
        frame, metric, args.seed + offset, args.bootstrap_repetitions
    )
    return value, low, high


def summarize(frame, args):
    rows = []
    metrics = (
        "selected_honest_support", "strict_edge", "pair_tie",
        "artifact_direct_edge_FT", "same_scc_rate",
        "final_F_before_T_rate", "strict_edge_and_final_rate",
    )
    grouped = frame.groupby(
        ["dataset", "actual_corruptions", "strategy"], sort=False
    )
    for group_index, ((dataset, corruptions, strategy), group) in enumerate(grouped):
        k = int(group.k_observers.iloc[0])
        b = int(group.b_fake_reports.iloc[0])
        row = {
            "dataset": dataset,
            "actual_corruptions": corruptions,
            "strategy": strategy,
            "k_observers": k,
            "b_fake_reports": b,
            "pair_trials": len(group),
            "final_trials": int(group.final_F_before_T_rate.notna().sum()),
            "deployments": group.deployment.nunique(),
        }
        for metric_index, metric in enumerate(metrics):
            selected = group[np.isfinite(group[metric])]
            bootstrap_seed = matched.stable_seed(
                args.seed, "corruption-budget-bootstrap", dataset,
                int(corruptions), k, b, metric,
            )
            value, low, high = bootstrap(
                selected, metric, args, bootstrap_seed - args.seed,
            )
            row[metric] = value
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gains(summary):
    """Report how observer and report capabilities compose at fixed c."""
    rows = []
    for (dataset, corruptions), group in summary.groupby(
        ["dataset", "actual_corruptions"], sort=False
    ):
        arms = group.set_index("strategy")
        for metric in (
            "strict_edge", "final_F_before_T_rate",
            "strict_edge_and_final_rate",
        ):
            observers = arms.loc["observers_truthful", metric]
            reports = arms.loc["single_observer_reports", metric]
            one_report = arms.loc["observers_one_report", metric]
            joint = arms.loc["joint", metric]
            rows.append({
                "dataset": dataset,
                "actual_corruptions": corruptions,
                "metric": metric,
                "observers_truthful": observers,
                "single_observer_reports": reports,
                "observers_one_report": one_report,
                "joint": joint,
                "joint_minus_best_one_sided": joint - max(observers, reports),
                "one_report_marginal_at_c_observers": one_report - observers,
                "remaining_report_marginal": joint - one_report,
                "observer_union_marginal_at_c_reports": joint - reports,
            })
    return pd.DataFrame(rows)


def percent_ci(row, metric):
    return (
        f"{100 * row[metric]:.1f}% "
        f"[{100 * row[f'{metric}_ci95_low']:.1f}%, "
        f"{100 * row[f'{metric}_ci95_high']:.1f}%]"
    )


def write_report(summary, gains, output_dir, args):
    indexed = summary.set_index(
        ["dataset", "actual_corruptions", "strategy"]
    )
    gain_indexed = gains.set_index(
        ["dataset", "actual_corruptions", "metric"]
    )
    lines = [
        "# Corruption-Budget Composition Ablation",
        "",
        "The protocol remains `n=21, f=5`. Actual corruptions are "
        "`c=1,...,5`; an arm with `k` active observers and `b` pair-swap "
        "reporters requires `max(k,b)=c` corrupted replicas. For each dataset, "
        "deployment, and c, the c locations are greedily trained on history for "
        "the joint `(c,c)` objective and then frozen. All four arms reuse that "
        "placement, execution sample, victim, workload, reporter set, and public "
        "Unspool tape.",
        "",
        "The first observer is the first location in the frozen greedy order, "
        "not the execution-time fastest or ex-post best source. At c=1, the "
        "three report-bearing labels all denote the same `(1,1)` strategy and "
        "must produce identical rows.",
        "",
        "## Main result",
        "",
        f"At `c=5`, the joint strict-edge rate is "
        f"{100 * indexed.loc[('aws', 5, 'joint'), 'strict_edge']:.1f}% on "
        f"AWS and {100 * indexed.loc[('wp', 5, 'joint'), 'strict_edge']:.1f}% "
        "on WP. Relative to the stronger of the two one-sided arms `(c,0)` "
        "and `(1,c)`, this is a matched gain of "
        f"{100 * gain_indexed.loc[('aws', 5, 'strict_edge'), 'joint_minus_best_one_sided']:.1f} "
        f"and {100 * gain_indexed.loc[('wp', 5, 'strict_edge'), 'joint_minus_best_one_sided']:.1f} "
        "percentage points, respectively. WP exceeds 50% strict-edge success "
        "at `c=3`; AWS reaches that level at `c=5`.",
        "",
        "The `c=5` rows `(5,0)`, `(1,5)`, and `(5,5)` exactly replay the "
        "previous composition artifact. Other budgets retrain placement for "
        "their own joint `(c,c)` threshold and freeze it across the four arms; "
        "they are not execution-time best-source selections. In particular, "
        "the matched `c=1` truthful row need not equal an independently trained "
        "truthful-singleton experiment.",
        "",
        "| Dataset | c | Strategy | (k,b) | Honest support | Strict edge | "
        "Pair tie | Same SCC | Final F<T | Strict edge and final |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report_order = {name: index for index, (name, _) in enumerate(STRATEGIES)}
    report_rows = summary.assign(
        strategy_order=summary.strategy.map(report_order)
    ).sort_values(["dataset", "actual_corruptions", "strategy_order"])
    for _, row in report_rows.iterrows():
        lines.append(
            f"| {row.dataset.upper()} | {int(row.actual_corruptions)} | "
            f"{row.strategy} | ({int(row.k_observers)},{int(row.b_fake_reports)}) "
            f"| {row.selected_honest_support:.2f} | "
            f"{percent_ci(row, 'strict_edge')} | {percent_ci(row, 'pair_tie')} "
            f"| {percent_ci(row, 'same_scc_rate')} | "
            f"{percent_ci(row, 'final_F_before_T_rate')} | "
            f"{percent_ci(row, 'strict_edge_and_final_rate')} |"
        )
    lines.extend([
        "",
        "## Composition gain",
        "",
        "The next table subtracts the stronger one-sided arm, "
        "`max((c,0),(1,c))`, from the joint `(c,c)` arm. A positive value "
        "is the matched gain that cannot be attributed to observer count or "
        "report count alone.",
        "",
        "| Dataset | c | Strict-edge gain | Final-order gain | "
        "Strict-and-final gain |",
        "|---|---:|---:|---:|---:|",
    ])
    for (dataset, corruptions), group in gains.groupby(
        ["dataset", "actual_corruptions"], sort=False
    ):
        values = group.set_index("metric").joint_minus_best_one_sided
        lines.append(
            f"| {dataset.upper()} | {int(corruptions)} | "
            f"{100 * values['strict_edge']:+.1f} pp | "
            f"{100 * values['final_F_before_T_rate']:+.1f} pp | "
            f"{100 * values['strict_edge_and_final_rate']:+.1f} pp |"
        )
    lines.extend([
        "",
        "Strict edge is the primary composition metric. Raw final order is "
        "reported separately because 300 background transactions can place T "
        "and F in one Condorcet SCC, after which public Unspool randomness may "
        "reverse either direct edge.",
        "",
        f"Each direct-edge point contains {args.deployments * args.trials_per_deployment} "
        "trials. Each final-order point contains "
        f"{args.deployments * args.final_trials_per_deployment} workloads and "
        f"averages {args.public_repetitions} public Unspool permutations per "
        "workload. Intervals are nominal 95% hierarchical bootstrap intervals.",
        "",
    ])
    with open(os.path.join(output_dir, "RESULTS.md"), "w") as file:
        file.write("\n".join(lines))


def plot(summary, output_dir):
    colors = {
        "observers_truthful": "#1F6F8B",
        "single_observer_reports": "#D97706",
        "observers_one_report": "#6B8E23",
        "joint": "#9B3A3A",
    }
    labels = {
        "observers_truthful": "(c,0): observers",
        "single_observer_reports": "(1,c): reports",
        "observers_one_report": "(c,1): observers + 1 report",
        "joint": "(c,c): joint",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for row_index, dataset in enumerate(("aws", "wp")):
        for column, metric in enumerate(("strict_edge", "final_F_before_T_rate")):
            ax = axes[row_index, column]
            for strategy, group in summary[
                summary.dataset == dataset
            ].groupby("strategy", sort=False):
                group = group.sort_values("actual_corruptions")
                y = group[metric].to_numpy()
                low = group[f"{metric}_ci95_low"].to_numpy()
                high = group[f"{metric}_ci95_high"].to_numpy()
                ax.errorbar(
                    group.actual_corruptions, y,
                    yerr=np.vstack((y - low, high - y)),
                    label=labels[strategy], color=colors[strategy], marker="o",
                    linewidth=2, capsize=3,
                )
            ax.set_title(
                f"{dataset.upper()}: "
                + ("strict F->T edge" if column == 0 else "final F<T")
            )
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_yticklabels([f"{value:.0%}" for value in np.linspace(0, 1, 5)])
            ax.set_xticks(range(1, 6))
            ax.grid(axis="y", alpha=0.25)
            if row_index == 1:
                ax.set_xlabel("Actual corruptions c")
            if column == 0:
                ax.set_ylabel("Trial rate")
    axes[0, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            os.path.join(output_dir, f"corruption_budget_ablation.{suffix}"),
            dpi=240, bbox_inches="tight",
        )
    plt.close(fig)


def run(args):
    if not 1 <= args.workers <= 5:
        raise ValueError("workers must be between 1 and 5")
    if args.final_trials_per_deployment > args.trials_per_deployment:
        raise ValueError("final trial count exceeds pair trial count")
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [
        (ping_file, deployment, corruptions, vars(args))
        for ping_file in args.ping_files
        for deployment in range(args.deployments)
        for corruptions in args.corruptions
    ]
    results = []
    if args.workers == 1:
        iterator = map(run_case, tasks)
        for result in tqdm(iterator, total=len(tasks), desc="budget cases"):
            results.append(result)
    else:
        with Pool(args.workers) as pool:
            for result in tqdm(
                pool.imap(run_case, tasks), total=len(tasks),
                desc="budget cases",
            ):
                results.append(result)
    frame = pd.DataFrame([row for rows, _ in results for row in rows])
    manifests = pd.DataFrame([manifest for _, manifest in results])
    summary = summarize(frame, args)
    gains = summarize_gains(summary)
    frame.to_csv(
        os.path.join(args.output_dir, "trial_results.csv.gz"),
        index=False, compression="gzip",
    )
    manifests.to_csv(
        os.path.join(args.output_dir, "deployment_manifest.csv"), index=False
    )
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    gains.to_csv(
        os.path.join(args.output_dir, "composition_gain.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump({
            "arguments": vars(args),
            "protocol": {"n": 21, "f": 5, "reports": 16, "threshold": 6},
            "actual_corruption_rule": "max(k,b)=c; observer and reporter roles overlap",
            "placement": (
                "arm-matched within c; history-trained for joint (c,c), all trace "
                "locations eligible, then frozen before execution"
            ),
            "report_behavior": (
                "only the first b corrupted identities swap T/F labels; honest "
                "and non-reporting corrupted identities retain received order"
            ),
            "confidence_interval": (
                "nominal 95% percentile hierarchical bootstrap; deployments then "
                "trials within deployment"
            ),
            "finalizer": "shared Python Themis graph/SCC/Unspool path validated elsewhere against C++",
        }, file, indent=2)
    write_report(summary, gains, args.output_dir, args)
    plot(summary, args.output_dir)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping_files", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deployments", type=int, default=20)
    parser.add_argument("--trials_per_deployment", type=int, default=100)
    parser.add_argument("--final_trials_per_deployment", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--corruptions", nargs="+", type=int, default=list(range(1, 6)),
        choices=range(1, 6),
    )
    parser.add_argument(
        "--replica_layout", choices=["latency-clustered"],
        default="latency-clustered",
    )
    parser.add_argument("--rho_ms", type=float, default=0.0)
    parser.add_argument("--background_txs", type=int, default=300)
    parser.add_argument("--public_repetitions", type=int, default=20)
    parser.add_argument("--conformance_trials_per_deployment", type=int, default=0)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
