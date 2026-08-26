#!/usr/bin/env python3
"""Matched ablation of corrupted-observer composition and Byzantine reports."""

import argparse
import itertools
import json
import math
import os
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate_corruption_aware_guard as guard
import evaluate_corruption_guard_background as background
import matched_multientry_evaluation as matched


class SampleWindow:
    """Expose sample-isolated CSV traces through the WindowTrace interface."""

    def __init__(self, trace):
        self.trace = trace
        self.locations = trace.locations
        self.index = {
            location: position for position, location in enumerate(self.locations)
        }

    def sample(self, _epoch, source, destination, seed, *parts):
        return self.trace.sample(
            "execution", source, destination, seed, *parts
        )


def load_window(trace, ping_file):
    if trace.is_hourly_snapshot:
        return guard.WindowTrace(ping_file)
    return SampleWindow(trace)


def pair_snapshot(trace, window, deployment, deployment_id, trial, args):
    if trace.is_hourly_snapshot:
        return guard.pair_snapshot(
            trace, window, deployment, deployment_id, trial, args
        )

    replicas = deployment["replicas"]
    victim = matched.stable_rng(
        args.seed, trace.dataset, deployment_id, trial, "victim"
    ).choice(deployment["clients"])
    t_arrivals = matched.T_GEN_TIME + np.asarray([
        trace.sample(
            "execution", victim, replica, args.seed,
            deployment_id, trial, "T", replica,
        )
        for replica in replicas
    ], dtype=np.float64)
    dep_args = deployment_args(args)
    corrupted = matched.corrupted_replica_indices(deployment, dep_args)
    direct = np.asarray([
        [
            t_arrivals[source_index]
            + args.rho_ms
            + trace.sample(
                "execution", replicas[source_index], replica, args.seed,
                deployment_id, trial, "attack-F",
                replicas[source_index], replica,
            )
            for replica in replicas
        ]
        for source_index in corrupted
    ], dtype=np.float64)
    return {
        "epoch": None,
        "hour": "odd-sample execution partition",
        "victim": victim,
        "t_arrivals": t_arrivals,
        "attack_f_arrivals": np.min(direct, axis=0),
        "corrupted": corrupted,
    }


def deployment_args(args, byzantine_reports=5):
    return SimpleNamespace(
        seed=args.seed,
        num_nodes=21,
        num_clients=15,
        entry_counts=[5],
        replica_layout=getattr(args, "replica_layout", "latency-clustered"),
        replica_regions=getattr(args, "replica_regions", 5),
        observation_model="corrupted-replica-direct",
        corrupted_observers=5,
        corrupted_placement="history-optimized-global",
        allow_corrupted_region_colocation=True,
        placement_victim_scope="all-locations",
        relay_entry_count=getattr(args, "location_pool_size", 220),
        byzantine_reports=byzantine_reports,
        rho_ms=args.rho_ms,
        training_snapshots=24,
    )


def singleton_arrivals(window, deployment, snapshot, deployment_id, trial, args):
    """Reconstruct each causal observation-plus-delivery vector separately."""
    replicas = deployment["replicas"]
    values = {}
    for corrupted_index in snapshot["corrupted"]:
        source = replicas[corrupted_index]
        observed = snapshot["t_arrivals"][corrupted_index]
        values[corrupted_index] = np.asarray([
            observed
            + args.rho_ms
            + window.sample(
                snapshot["epoch"], source, replica, args.seed,
                deployment_id, trial, "attack-F", source, replica,
            )
            for replica in replicas
        ], dtype=np.float64)
    reconstructed_union = np.min(np.stack(list(values.values())), axis=0)
    if not np.allclose(
        reconstructed_union, snapshot["attack_f_arrivals"], rtol=0, atol=1e-9
    ):
        raise AssertionError("singleton union does not reproduce headline attack")
    return values


def evaluate_pair(deployment, snapshot, f_arrivals, report_behavior):
    reporters = np.asarray(deployment["reporter_order"][:16], dtype=int)
    corrupted = set(snapshot["corrupted"])
    corrupted_mask = np.asarray(
        [index in corrupted for index in reporters], dtype=bool
    )
    selected_t = snapshot["t_arrivals"][reporters]
    selected_f = np.asarray(f_arrivals, dtype=np.float64)[reporters]
    # Full local orderings append T before F and use a stable time sort. Mirror
    # that deterministic T-before-F policy for exact arrival ties at pair level.
    selected_f = guard.encode_t_before_f_ties(selected_t, selected_f)
    report_t, report_f = guard.apply_report_behavior(
        selected_t, selected_f, corrupted_mask, report_behavior
    )
    w_ft, w_tf, strict_edge = guard.strict_edge(report_t, report_f)
    honest_winners = {
        int(index) for position, index in enumerate(reporters)
        if not corrupted_mask[position]
        and selected_f[position] < selected_t[position] - guard.EPS_MS
    }
    all_winners = {
        int(index) for position, index in enumerate(reporters)
        if selected_f[position] < selected_t[position] - guard.EPS_MS
    }
    return {
        "w_FT": w_ft,
        "w_TF": w_tf,
        "strict_edge": strict_edge,
        "selected_honest_support": len(honest_winners),
        "selected_receive_order_support": len(all_winners),
        "honest_winners": honest_winners,
        "all_winners": all_winners,
    }


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else math.nan


def arm_row(common, arm, result, report_behavior, source_count,
            source_id="", source_location=""):
    return {
        **common,
        "arm": arm,
        "report_behavior": report_behavior,
        "source_count": source_count,
        "source_id": source_id,
        "source_location": source_location,
        "w_FT": result["w_FT"],
        "w_TF": result["w_TF"],
        "selected_honest_support": result["selected_honest_support"],
        "selected_receive_order_support": result[
            "selected_receive_order_support"
        ],
        "strict_edge": result["strict_edge"],
    }


def run_deployment(task):
    deployment_id, args_dict = task
    args = SimpleNamespace(**args_dict)
    trace = matched.LatencyTrace(args.ping_file, "half", 0.15)
    window = load_window(trace, args.ping_file)
    dep_args = deployment_args(args)
    deployment = matched.make_deployment(trace, deployment_id, dep_args)
    truthful_dep_args = deployment_args(args, byzantine_reports=0)
    truthful_singleton_deployment = matched.make_deployment(
        trace, deployment_id, truthful_dep_args
    )
    corrupted = matched.corrupted_replica_indices(deployment, dep_args)
    if len(corrupted) != 5:
        raise AssertionError("composition experiment requires five observers")
    reporters = deployment["reporter_order"][:16]
    if not set(corrupted).issubset(reporters):
        raise AssertionError("attacker-favorable report set changed")

    trial_rows = []
    arm_rows = []
    singleton_rows = []
    for trial in range(args.trials_per_deployment):
        snapshot = pair_snapshot(
            trace, window, deployment, deployment_id, trial, args
        )
        truthful_singleton_snapshot = pair_snapshot(
            trace, window, truthful_singleton_deployment,
            deployment_id, trial, args,
        )
        singles = singleton_arrivals(
            window, deployment, snapshot, deployment_id, trial, args
        )
        truthful_singleton_arrivals = singleton_arrivals(
            window, truthful_singleton_deployment,
            truthful_singleton_snapshot, deployment_id, trial, args,
        )
        common = {
            "dataset": trace.dataset,
            "deployment": deployment_id,
            "trial": trial,
            "victim": snapshot["victim"],
            "execution_snapshot_hour": snapshot["hour"],
            "rho_ms": args.rho_ms,
            "reporter_ids": ";".join(str(value) for value in reporters),
            "corrupted_ids": ";".join(str(value) for value in corrupted),
            "corrupted_locations": ";".join(
                deployment["replicas"][index] for index in corrupted
            ),
        }

        singleton_results = {}
        singleton_honest_results = {}
        for corrupted_index, arrivals in singles.items():
            result = evaluate_pair(
                deployment, snapshot, arrivals, "pair-swap"
            )
            honest_result = evaluate_pair(
                deployment, snapshot, arrivals, "honest"
            )
            singleton_results[corrupted_index] = result
            singleton_honest_results[corrupted_index] = honest_result
            singleton_rows.append({
                **common,
                "observer_id": corrupted_index,
                "observer_location": deployment["replicas"][corrupted_index],
                "w_FT": result["w_FT"],
                "w_TF": result["w_TF"],
                "selected_honest_support": result[
                    "selected_honest_support"
                ],
                "strict_edge": result["strict_edge"],
                "honest_winner_ids": ";".join(
                    str(value) for value in sorted(result["honest_winners"])
                ),
            })

        best_index, best_result = max(
            singleton_results.items(),
            key=lambda item: (
                item[1]["strict_edge"],
                item[1]["selected_honest_support"],
                item[1]["w_FT"],
                -item[0],
            ),
        )
        history_singleton_index = corrupted[0]
        history_singleton_reports = singleton_results[history_singleton_index]
        truthful_singleton_index = truthful_singleton_snapshot["corrupted"][0]
        history_singleton_honest = evaluate_pair(
            truthful_singleton_deployment, truthful_singleton_snapshot,
            truthful_singleton_arrivals[truthful_singleton_index], "honest",
        )
        union_arrivals = np.min(np.stack(list(singles.values())), axis=0)
        observation_only = evaluate_pair(
            deployment, snapshot, union_arrivals, "honest"
        )
        joint = evaluate_pair(
            deployment, snapshot, union_arrivals, "pair-swap"
        )

        arm_rows.extend([
            arm_row(
                common, "history-singleton+honest",
                history_singleton_honest, "honest", 1,
                str(truthful_singleton_index),
                truthful_singleton_deployment["replicas"][
                    truthful_singleton_index
                ],
            ),
            arm_row(
                common, "history-singleton+reports",
                history_singleton_reports, "pair-swap", 1,
                str(history_singleton_index),
                deployment["replicas"][history_singleton_index],
            ),
            arm_row(
                common, "any-singleton+reports", best_result, "pair-swap", 1,
                str(best_index), deployment["replicas"][best_index],
            ),
            arm_row(
                common, "observer-union+honest", observation_only, "honest", 5,
            ),
            arm_row(common, "joint", joint, "pair-swap", 5),
        ])

        winner_sets = [
            result["honest_winners"] for result in singleton_results.values()
        ]
        winner_union = set().union(*winner_sets)
        pairwise = [
            jaccard(winner_sets[left], winner_sets[right])
            for left, right in itertools.combinations(range(len(winner_sets)), 2)
        ]
        finite_jaccard = [value for value in pairwise if math.isfinite(value)]
        max_singleton_support = max(
            result["selected_honest_support"]
            for result in singleton_results.values()
        )
        any_singleton_edge = max(
            result["strict_edge"] for result in singleton_results.values()
        )
        any_truthful_singleton_edge = max(
            result["strict_edge"]
            for result in singleton_honest_results.values()
        )
        strict_separation = int(
            joint["strict_edge"] == 1 and any_singleton_edge == 0
        )
        truthful_strict_separation = int(
            observation_only["strict_edge"] == 1
            and any_truthful_singleton_edge == 0
        )
        threshold_witness = int(
            max_singleton_support <= 3
            and joint["selected_honest_support"] >= 4
        )
        if strict_separation != threshold_witness:
            raise AssertionError(
                "Themis threshold witness and strict-edge separation diverged"
            )

        final = {
            "history_singleton_honest_artifact_direct_edge_FT": math.nan,
            "history_singleton_honest_artifact_direct_edge_TF": math.nan,
            "history_singleton_honest_final_F_before_T_rate": math.nan,
            "history_singleton_honest_strict_edge_and_final_rate": math.nan,
            "history_singleton_honest_same_scc_rate": math.nan,
            "history_singleton_reports_artifact_direct_edge_FT": math.nan,
            "history_singleton_reports_artifact_direct_edge_TF": math.nan,
            "history_singleton_reports_final_F_before_T_rate": math.nan,
            "history_singleton_reports_strict_edge_and_final_rate": math.nan,
            "history_singleton_reports_same_scc_rate": math.nan,
            "oracle_singleton_reports_artifact_direct_edge_FT": math.nan,
            "oracle_singleton_reports_artifact_direct_edge_TF": math.nan,
            "oracle_singleton_reports_final_F_before_T_rate": math.nan,
            "oracle_singleton_reports_strict_edge_and_final_rate": math.nan,
            "oracle_singleton_reports_same_scc_rate": math.nan,
            "observer_union_honest_artifact_direct_edge_FT": math.nan,
            "observer_union_honest_artifact_direct_edge_TF": math.nan,
            "observer_union_honest_final_F_before_T_rate": math.nan,
            "observer_union_honest_strict_edge_and_final_rate": math.nan,
            "observer_union_honest_same_scc_rate": math.nan,
            "joint_artifact_direct_edge_FT": math.nan,
            "joint_artifact_direct_edge_TF": math.nan,
            "joint_final_F_before_T_rate": math.nan,
            "joint_strict_edge_and_final_rate": math.nan,
            "strict_separation_and_final_rate": math.nan,
            "truthful_strict_separation_and_final_rate": math.nan,
            "joint_same_scc_rate": math.nan,
        }
        if trial < args.final_trials_per_deployment:
            _, _, background_arrivals = background.generate_background(
                window, snapshot, deployment, deployment_id, trial,
                args.background_txs, args.seed,
            )
            _, _, truthful_background_arrivals = background.generate_background(
                window, truthful_singleton_snapshot,
                truthful_singleton_deployment, deployment_id, trial,
                args.background_txs, args.seed,
            )
            final_specs = [
                (
                    "history_singleton_honest",
                    truthful_singleton_arrivals[truthful_singleton_index],
                    history_singleton_honest, "honest",
                    truthful_singleton_deployment,
                    truthful_singleton_snapshot,
                    truthful_background_arrivals,
                ),
                (
                    "history_singleton_reports",
                    singles[history_singleton_index],
                    history_singleton_reports, "pair-swap",
                    deployment, snapshot, background_arrivals,
                ),
                (
                    "oracle_singleton_reports", singles[best_index],
                    best_result, "pair-swap",
                    deployment, snapshot, background_arrivals,
                ),
                (
                    "observer_union_honest", union_arrivals,
                    observation_only, "honest",
                    deployment, snapshot, background_arrivals,
                ),
                (
                    "joint", union_arrivals, joint, "pair-swap",
                    deployment, snapshot, background_arrivals,
                ),
            ]
            final = {}
            for (
                final_name, final_arrivals, pair_result, report_behavior,
                final_deployment, source_snapshot, source_background
            ) in final_specs:
                final_snapshot = dict(source_snapshot)
                final_snapshot["attack_f_arrivals"] = final_arrivals
                final_result = background.evaluate_attack_unspool(
                    final_deployment, final_snapshot, source_background,
                    {
                        "w_FT": pair_result["w_FT"],
                        "w_TF": pair_result["w_TF"],
                        "deferred_edge": 0,
                    },
                    deployment_id, trial, args, report_behavior,
                )
                if (
                    pair_result["strict_edge"] == 1
                    and final_result["raw_direct_edge_FT"] != 1
                ):
                    raise AssertionError(
                        f"full graph dropped a strict F/T edge for {final_name}"
                    )
                raw_final = final_result["raw_final_F_before_T_rate"]
                final.update({
                    f"{final_name}_artifact_direct_edge_FT": final_result[
                        "raw_direct_edge_FT"
                    ],
                    f"{final_name}_artifact_direct_edge_TF": final_result[
                        "raw_direct_edge_TF"
                    ],
                    f"{final_name}_final_F_before_T_rate": raw_final,
                    f"{final_name}_strict_edge_and_final_rate": (
                        pair_result["strict_edge"] * raw_final
                    ),
                    f"{final_name}_same_scc_rate": final_result[
                        "raw_same_scc_rate"
                    ],
                })
            final["strict_separation_and_final_rate"] = (
                strict_separation * final["joint_final_F_before_T_rate"]
            )
            final["truthful_strict_separation_and_final_rate"] = (
                truthful_strict_separation
                * final["observer_union_honest_final_F_before_T_rate"]
            )

        trial_rows.append({
            **common,
            "best_singleton_id": best_index,
            "best_singleton_location": deployment["replicas"][best_index],
            "max_singleton_honest_support": max_singleton_support,
            "union_honest_support": joint["selected_honest_support"],
            "union_support_gain": (
                joint["selected_honest_support"] - max_singleton_support
            ),
            "winner_union_size": len(winner_union),
            "mean_pairwise_jaccard": (
                float(np.mean(finite_jaccard)) if finite_jaccard else math.nan
            ),
            "any_singleton_strict_edge": any_singleton_edge,
            "any_truthful_singleton_strict_edge": (
                any_truthful_singleton_edge
            ),
            "history_singleton_honest_strict_edge": (
                history_singleton_honest["strict_edge"]
            ),
            "history_singleton_reports_strict_edge": (
                history_singleton_reports["strict_edge"]
            ),
            "observation_only_strict_edge": observation_only["strict_edge"],
            "joint_strict_edge": joint["strict_edge"],
            "strict_edge_separation": strict_separation,
            "truthful_strict_edge_separation": truthful_strict_separation,
            "threshold_composition_witness": threshold_witness,
            **final,
        })
    return trial_rows, arm_rows, singleton_rows, {
        "dataset": trace.dataset,
        "deployment": deployment_id,
        "replicas": json.dumps(deployment["replicas"]),
        "clients": json.dumps(deployment["clients"]),
        "reporter_order": json.dumps(deployment["reporter_order"]),
        "corrupted_ids": json.dumps(corrupted),
        "corrupted_locations": json.dumps([
            deployment["replicas"][index] for index in corrupted
        ]),
        "history_singleton_id": corrupted[0],
        "history_singleton_location": deployment["replicas"][corrupted[0]],
        "truthful_singleton_id": (
            truthful_singleton_deployment["corrupted_replica_indices"][0]
        ),
        "truthful_singleton_location": truthful_singleton_deployment[
            "replicas"
        ][truthful_singleton_deployment["corrupted_replica_indices"][0]],
        "corrupted_placement_info": json.dumps(
            deployment["corrupted_placement_info"]
        ),
    }


def bootstrap_metric(frame, metric, args, offset):
    low, high = matched.hierarchical_bootstrap(
        frame, metric, args.seed + offset, args.bootstrap_repetitions
    )
    return float(frame[metric].mean()), low, high


def summarize_arms(arms, args):
    rows = []
    for arm_index, (arm, group) in enumerate(arms.groupby("arm", sort=False)):
        row = {
            "arm": arm,
            "trials": len(group),
            "deployments": group.deployment.nunique(),
            "report_behavior": group.report_behavior.iloc[0],
            "source_count": int(group.source_count.iloc[0]),
        }
        for metric_index, metric in enumerate(
            ["selected_honest_support", "w_FT", "strict_edge"]
        ):
            mean, low, high = bootstrap_metric(
                group, metric, args, 10000 + arm_index * 100 + metric_index
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_composition(trials, args):
    metrics = [
        "max_singleton_honest_support",
        "union_honest_support",
        "union_support_gain",
        "mean_pairwise_jaccard",
        "any_singleton_strict_edge",
        "any_truthful_singleton_strict_edge",
        "history_singleton_honest_strict_edge",
        "history_singleton_reports_strict_edge",
        "observation_only_strict_edge",
        "joint_strict_edge",
        "strict_edge_separation",
        "truthful_strict_edge_separation",
    ]
    rows = []
    for metric_index, metric in enumerate(metrics):
        group = trials[np.isfinite(trials[metric])]
        mean, low, high = bootstrap_metric(
            group, metric, args, 20000 + metric_index
        )
        rows.append({
            "metric": metric,
            "trials": len(group),
            "deployments": group.deployment.nunique(),
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        })
    final = trials[np.isfinite(trials.strict_separation_and_final_rate)]
    if len(final):
        for offset, metric in enumerate([
            "history_singleton_honest_artifact_direct_edge_FT",
            "history_singleton_honest_artifact_direct_edge_TF",
            "history_singleton_honest_final_F_before_T_rate",
            "history_singleton_honest_strict_edge_and_final_rate",
            "history_singleton_honest_same_scc_rate",
            "history_singleton_reports_artifact_direct_edge_FT",
            "history_singleton_reports_artifact_direct_edge_TF",
            "history_singleton_reports_final_F_before_T_rate",
            "history_singleton_reports_strict_edge_and_final_rate",
            "history_singleton_reports_same_scc_rate",
            "oracle_singleton_reports_artifact_direct_edge_FT",
            "oracle_singleton_reports_artifact_direct_edge_TF",
            "oracle_singleton_reports_final_F_before_T_rate",
            "oracle_singleton_reports_strict_edge_and_final_rate",
            "oracle_singleton_reports_same_scc_rate",
            "observer_union_honest_artifact_direct_edge_FT",
            "observer_union_honest_artifact_direct_edge_TF",
            "observer_union_honest_final_F_before_T_rate",
            "observer_union_honest_strict_edge_and_final_rate",
            "observer_union_honest_same_scc_rate",
            "joint_artifact_direct_edge_FT",
            "joint_artifact_direct_edge_TF",
            "joint_final_F_before_T_rate",
            "joint_strict_edge_and_final_rate",
            "strict_separation_and_final_rate",
            "truthful_strict_separation_and_final_rate",
            "joint_same_scc_rate",
        ]):
            mean, low, high = bootstrap_metric(
                final, metric, args, 21000 + offset
            )
            rows.append({
                "metric": metric,
                "trials": len(final),
                "deployments": final.deployment.nunique(),
                "mean": mean,
                "ci95_low": low,
                "ci95_high": high,
            })
    ratio_specs = [
        (
            "strict_separation_given_joint",
            "strict_edge_separation",
            "joint_strict_edge",
        ),
        (
            "joint_success_given_all_singletons_fail",
            "strict_edge_separation",
            "all_singletons_fail",
        ),
        (
            "truthful_separation_given_union",
            "truthful_strict_edge_separation",
            "observation_only_strict_edge",
        ),
        (
            "truthful_union_success_given_all_singletons_fail",
            "truthful_strict_edge_separation",
            "all_truthful_singletons_fail",
        ),
    ]
    ratio_frame = trials.assign(
        all_singletons_fail=1 - trials.any_singleton_strict_edge,
        all_truthful_singletons_fail=(
            1 - trials.any_truthful_singleton_strict_edge
        ),
    )
    for offset, (name, numerator, denominator) in enumerate(ratio_specs):
        denominator_sum = float(ratio_frame[denominator].sum())
        mean = (
            float(ratio_frame[numerator].sum() / denominator_sum)
            if denominator_sum else math.nan
        )
        low, high = guard.bootstrap_ratio(
            ratio_frame, numerator, denominator,
            args.seed + 22000 + offset, args.bootstrap_repetitions,
        )
        rows.append({
            "metric": name,
            "trials": len(ratio_frame),
            "deployments": ratio_frame.deployment.nunique(),
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
        })
    return pd.DataFrame(rows)


def percent(value):
    return f"{100 * value:.1f}%"


def write_report(arm_summary, composition, output_dir, args):
    arms = arm_summary.set_index("arm")
    metrics = composition.set_index("metric")
    lines = [
        "# Corrupted-Observer Composition Ablation",
        "",
        f"This matched {args.dataset.upper()} experiment fixes each deployment, "
        "victim, execution sample, reporter set, and random seed across all "
        "counterfactual arms. The committee is `n=21, f=5`, the selected set "
        "contains 16 complete reports including all five corrupted replicas, "
        f"the location pool has {args.location_pool_size} sites, replica layout "
        f"is `{args.replica_layout}`, "
        f"dissemination is direct/plain, and `rho={args.rho_ms:g} ms`.",
        "",
        "Each operational `history-singleton` is selected independently on "
        "history for its report regime and frozen before execution. The truthful "
        "and five-swap rows may therefore select different locations. "
        "`any-singleton+reports` "
        "is a stronger execution-oracle event: it succeeds if any one of the "
        "five individual observers succeeds, but it never combines their "
        "transaction-plane paths. In every `+reports` arm, the same five "
        "corrupted identities may submit pair-swap reports; the other four are "
        "report forgers but cannot source F in a singleton replay. "
        "`observer-union+honest` uses all five transaction-plane paths without "
        "report forgery, and `joint` enables both capabilities.",
        "",
        "A pure-report sanity check fixes every non-Byzantine report to T<F. "
        "The five Byzantine reports then give `w(F,T)=5`, below both the support "
        "floor and the reverse weight `w(T,F)=11`, so no strict edge forms.",
        "",
        "## Direct-edge result",
        "",
        "| Arm | Selected honest F<T | Strict F->T edge (95% CI) |",
        "|---|---:|---:|",
    ]
    for arm in [
        "history-singleton+honest", "history-singleton+reports",
        "any-singleton+reports",
        "observer-union+honest", "joint",
    ]:
        row = arms.loc[arm]
        lines.append(
            f"| {arm} | {row.selected_honest_support_mean:.2f} | "
            f"{percent(row.strict_edge_mean)} "
            f"[{percent(row.strict_edge_ci95_low)}, "
            f"{percent(row.strict_edge_ci95_high)}] |"
        )
    separation = metrics.loc["strict_edge_separation"]
    truthful_separation = metrics.loc["truthful_strict_edge_separation"]
    conditional = metrics.loc["strict_separation_given_joint"]
    rescue = metrics.loc["joint_success_given_all_singletons_fail"]
    gain = metrics.loc["union_support_gain"]
    jaccard_row = metrics.loc["mean_pairwise_jaccard"]
    lines.extend([
        "",
        "## Composition witness",
        "",
        f"- Strict separation occurs in {percent(separation['mean'])} of trials "
        f"[{percent(separation.ci95_low)}, {percent(separation.ci95_high)}]: "
        "the joint arm creates a strict edge while every individual observer "
        "fails.",
        f"- Equivalently, multi-observer composition is necessary for "
        f"{percent(conditional['mean'])} of Joint strict-edge successes "
        f"[{percent(conditional.ci95_low)}, "
        f"{percent(conditional.ci95_high)}]. Among trials where every singleton "
        f"fails, the union crosses the threshold in {percent(rescue['mean'])} "
        f"[{percent(rescue.ci95_low)}, {percent(rescue.ci95_high)}].",
        f"- The observer union adds {gain['mean']:.2f} selected honest winners "
        f"over the strongest singleton on average "
        f"[{gain.ci95_low:.2f}, {gain.ci95_high:.2f}].",
        f"- Mean pairwise singleton winner-set Jaccard is "
        f"{jaccard_row['mean']:.3f} "
        f"[{jaccard_row.ci95_low:.3f}, {jaccard_row.ci95_high:.3f}].",
        "- `any-singleton+reports` deliberately retains all five Byzantine "
        "reports. This holds report-plane power constant and tests whether "
        "recipient-specific composition of multiple causal delivery paths is "
        "needed; it is not a one-corruption adversary.",
        f"- With truthful reports, observer-path separation occurs in "
        f"{percent(truthful_separation['mean'])} of trials "
        f"[{percent(truthful_separation.ci95_low)}, "
        f"{percent(truthful_separation.ci95_high)}].",
    ])
    if "strict_separation_and_final_rate" in metrics.index:
        final = metrics.loc["strict_separation_and_final_rate"]
        workload_rows = [
            ("history-fixed singleton, truthful", "history_singleton_honest"),
            ("history-fixed singleton, 5 swaps", "history_singleton_reports"),
            ("per-trial oracle singleton, 5 swaps", "oracle_singleton_reports"),
            ("observer union, truthful", "observer_union_honest"),
            ("Joint", "joint"),
        ]
        lines.extend([
            "",
            "## Full-workload auxiliary result",
            "",
            f"The first {args.final_trials_per_deployment} trials of each "
            f"deployment are replayed with {args.background_txs} Poisson "
            f"background transactions and {args.public_repetitions} public "
            "Unspool repetitions. Artifact direct edges include the original "
            "public pair-order tie handling; `strict edge and final` uses only "
            "the tie-independent `w(F,T)>w(T,F)` predicate.",
            "",
            "| Full-workload arm | Artifact F->T edge | Artifact T->F edge | "
            "Final F<T | Strict edge and final F<T |",
            "|---|---:|---:|---:|---:|",
        ])
        for label, prefix in workload_rows:
            edge_ft = metrics.loc[f"{prefix}_artifact_direct_edge_FT"]
            edge_tf = metrics.loc[f"{prefix}_artifact_direct_edge_TF"]
            final_ft = metrics.loc[f"{prefix}_final_F_before_T_rate"]
            strict_final = metrics.loc[
                f"{prefix}_strict_edge_and_final_rate"
            ]
            lines.append(
                f"| {label} | {percent(edge_ft['mean'])} | "
                f"{percent(edge_tf['mean'])} | {percent(final_ft['mean'])} | "
                f"{percent(strict_final['mean'])} |"
            )
        lines.extend([
            "",
            f"Strict singleton-union separation together with final `F<T` "
            f"occurs at rate {percent(final['mean'])} "
            f"[{percent(final.ci95_low)}, {percent(final.ci95_high)}].",
        ])
    lines.extend([
        "",
        "The strict-separation result is the primary causal-attribution metric. "
        "It is exactly checked against the Themis threshold witness "
        "`max singleton honest support <= 3` and "
        "`union honest support >= 4`. Final-order rates are secondary because "
        "SCC formation and public Unspool randomness can affect output after "
        "the direct edge has been formed.",
    ])
    with open(os.path.join(output_dir, "RESULTS.md"), "w") as file:
        file.write("\n".join(lines) + "\n")


def run(args):
    if args.workers < 1 or args.workers > 5:
        raise ValueError("workers must be between 1 and 5")
    if args.final_trials_per_deployment > args.trials_per_deployment:
        raise ValueError("final trial count exceeds direct-edge trial count")
    trace = matched.LatencyTrace(args.ping_file, "half", 0.15)
    args.dataset = trace.dataset
    args.location_pool_size = len(trace.locations)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "unspool-conformance"), exist_ok=True)
    sanity_t = np.zeros(16, dtype=np.float64)
    sanity_f = np.ones(16, dtype=np.float64)
    sanity_mask = np.zeros(16, dtype=bool)
    sanity_mask[:5] = True
    sanity_report_t, sanity_report_f = guard.apply_report_behavior(
        sanity_t, sanity_f, sanity_mask, "pair-swap"
    )
    sanity_w_ft, sanity_w_tf, sanity_edge = guard.strict_edge(
        sanity_report_t, sanity_report_f
    )
    if (sanity_w_ft, sanity_w_tf, sanity_edge) != (5, 11, 0):
        raise AssertionError("pure Byzantine reports unexpectedly form an edge")
    tasks = [(deployment, vars(args)) for deployment in range(args.deployments)]
    results = []
    if args.workers == 1:
        iterator = map(run_deployment, tasks)
        for value in tqdm(iterator, total=len(tasks), desc="deployments"):
            results.append(value)
    else:
        with Pool(args.workers) as pool:
            for value in tqdm(
                pool.imap(run_deployment, tasks), total=len(tasks),
                desc="deployments",
            ):
                results.append(value)

    trials = pd.DataFrame([
        row for trial_rows, _, _, _ in results for row in trial_rows
    ])
    arms = pd.DataFrame([
        row for _, arm_rows, _, _ in results for row in arm_rows
    ])
    singletons = pd.DataFrame([
        row for _, _, singleton_rows, _ in results for row in singleton_rows
    ])
    manifests = pd.DataFrame([value[3] for value in results])
    arm_summary = summarize_arms(arms, args)
    composition = summarize_composition(trials, args)

    trials.to_csv(os.path.join(args.output_dir, "trial_results.csv"), index=False)
    arms.to_csv(os.path.join(args.output_dir, "arm_results.csv"), index=False)
    singletons.to_csv(
        os.path.join(args.output_dir, "singleton_results.csv.gz"), index=False,
        compression="gzip",
    )
    manifests.to_csv(
        os.path.join(args.output_dir, "deployment_manifest.csv"), index=False
    )
    arm_summary.to_csv(
        os.path.join(args.output_dir, "arm_summary.csv"), index=False
    )
    composition.to_csv(
        os.path.join(args.output_dir, "composition_summary.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump({
            "arguments": vars(args),
            "confidence_interval": (
                "95% percentile hierarchical bootstrap; resample deployments, "
                "then trials within deployments"
            ),
            "strict_separation": (
                "joint strict F->T edge and no strict F->T edge from any of the "
                "five individual corrupted observers under the same five "
                "pair-swap reports"
            ),
            "operational_singleton_selection": (
                "independent history optimization for truthful (b=0) and "
                "five-swap (b=5) report regimes; frozen before execution"
            ),
            "pure_report_sanity": {
                "w_FT": sanity_w_ft,
                "w_TF": sanity_w_tf,
                "strict_edge": sanity_edge,
                "meaning": (
                    "five forged reports with zero honest F<T support cannot "
                    "form a strict edge"
                ),
            },
            "latency_model": (
                trace.snapshot_metadata.get(
                    "latency_model",
                    "coherent odd-hour WP snapshot with truncated-normal path "
                    "sampling",
                )
                if trace.is_hourly_snapshot else
                "AWS sample isolation: even-indexed RTT samples for placement "
                "and odd-indexed RTT samples for execution; RTT/2"
            ),
        }, file, indent=2)
    write_report(arm_summary, composition, args.output_dir, args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ping_file",
        default="./simulations/data-ping/wp-hourly-2020-07-19-20.npz",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "./simulations/data-causality-comparison/"
            "observer-composition-ablation00"
        ),
    )
    parser.add_argument("--deployments", type=int, default=20)
    parser.add_argument("--trials_per_deployment", type=int, default=100)
    parser.add_argument("--final_trials_per_deployment", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--replica_layout",
        choices=["latency-clustered", "global-random"],
        default="latency-clustered",
    )
    parser.add_argument("--replica_regions", type=int, default=5)
    parser.add_argument("--rho_ms", type=float, default=0.0)
    parser.add_argument("--rho_floor_ms", type=float, default=0.0)
    parser.add_argument("--bound_mode", default="same-hour")
    parser.add_argument("--background_txs", type=int, default=300)
    parser.add_argument("--public_repetitions", type=int, default=20)
    parser.add_argument("--conformance_trials_per_deployment", type=int, default=0)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
