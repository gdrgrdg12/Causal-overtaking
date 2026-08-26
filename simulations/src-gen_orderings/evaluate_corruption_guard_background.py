#!/usr/bin/env python3
"""Evaluate the corruption-aware guard with 300 Poisson background txs."""

import argparse
import json
import math
import os
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate_corruption_aware_guard as guard
import matched_multientry_evaluation as matched


COMPONENT_POLICIES = (
    "no-guard",
    "profile-only",
    "capacity-only",
    "profile+trace-bound",
    "carry-every-reverse-edge",
)


def policy_slug(policy):
    return policy.replace("+", "_plus_").replace("-", "_")


def generate_background(window, snapshot, deployment, deployment_id, trial,
                        count, seed):
    clients = deployment["clients"]
    replicas = deployment["replicas"]
    rng = np.random.default_rng(matched.stable_seed(
        seed, deployment.get("dataset", "wp"), deployment_id, trial,
        "background"
    ))
    per_client = np.full(len(clients), count // len(clients), dtype=int)
    per_client[:count % len(clients)] += 1
    releases = []
    sources = []
    for client_index, client_count in enumerate(per_client):
        if client_count == 0:
            continue
        client_releases = rng.exponential(
            1000.0 / client_count, client_count
        ).cumsum()
        releases.extend(client_releases)
        sources.extend([clients[client_index]] * client_count)
    releases = np.asarray(releases, dtype=np.float64)
    arrivals = np.empty((count, len(replicas)), dtype=np.float64)
    for tx_index, (source, release) in enumerate(zip(sources, releases)):
        for replica_index, replica in enumerate(replicas):
            arrivals[tx_index, replica_index] = (
                release
                + window.sample(
                    snapshot["epoch"], source, replica, seed,
                    deployment_id, trial, "background", tx_index, replica,
                )
            )
    return releases, sources, arrivals


def generate_section4_background(trace, deployment, deployment_id, trial,
                                 count, seed):
    """Replay the Section IV Poisson workload and execution sample keys."""
    clients = deployment["clients"]
    replicas = deployment["replicas"]
    rng = np.random.default_rng(matched.stable_seed(
        seed, deployment["dataset"], deployment_id, trial, "background"
    ))
    assignments = np.arange(count) % len(clients)
    rng.shuffle(assignments)
    releases = rng.exponential(
        matched.EPOCH_MS / max(count, 1), count
    ).cumsum()
    sources = [clients[int(index)] for index in assignments]
    arrivals = np.empty((count, len(replicas)), dtype=np.float64)
    for tx_index, (source, release) in enumerate(zip(sources, releases)):
        tx = f"B{tx_index:03d}"
        for replica_index, replica in enumerate(replicas):
            arrivals[tx_index, replica_index] = (
                release
                + trace.sample(
                    "execution", source, replica, seed,
                    "trial", deployment_id, trial, "background", tx, replica,
                )
            )
    return releases, sources, arrivals


def apply_cross_epoch_admission(raw_outcomes, deferred):
    """Return pair outcomes after an admission-only next-epoch carryover.

    A deferred F is absent from the epoch that contains T. It is not represented
    by synthetic same-epoch edges; after that epoch finalizes, ledger epoch
    order directly fixes T before F.
    """
    if not deferred:
        return [dict(outcome) for outcome in raw_outcomes]
    return [
        {"final_ft": 0, "same_scc": 0, "ft_scc_size": 0}
        for _ in raw_outcomes
    ]


def evaluate_attack_unspool(deployment, snapshot, arrivals, attack,
                            deployment_id, trial, args,
                            report_behavior="pair-swap",
                            byzantine_indices=None):
    """Run the complete Themis graph and finalizer on the attack workload."""
    reporters = deployment["reporter_order"][:16]
    orderings = []
    for replica_index in range(len(deployment["replicas"])):
        report = [
            (f"B{index:03d}", arrivals[index, replica_index])
            for index in range(len(arrivals))
        ]
        report.extend([
            ("T", snapshot["t_arrivals"][replica_index]),
            ("F", snapshot["attack_f_arrivals"][replica_index]),
        ])
        report.sort(key=lambda item: item[1])
        orderings.append(report)
    if report_behavior == "pair-swap":
        forged_indices = (
            snapshot["corrupted"]
            if byzantine_indices is None
            else byzantine_indices
        )
        for replica_index in forged_indices:
            orderings[replica_index] = matched.forge_byzantine_report(
                orderings[replica_index], "pair-swap"
            )
    elif report_behavior != "honest":
        raise ValueError(report_behavior)

    transactions = {
        tx for reporter in reporters for tx, _ in orderings[reporter]
    }
    pair_order = sorted(
        transactions,
        key=lambda tx: matched.stable_seed(
            args.seed, deployment.get("dataset", "wp"), deployment_id, trial,
            "pair-processing-order", tx,
        ),
    )
    tx_list, weights, raw_adjacency = matched.build_graph(
        orderings, reporters, 6, pair_order
    )
    f_index = tx_list.index("F")
    t_index = tx_list.index("T")
    if int(weights[f_index, t_index]) != int(attack["w_FT"]):
        raise AssertionError("full-workload F/T forward weight mismatch")
    if int(weights[t_index, f_index]) != int(attack["w_TF"]):
        raise AssertionError("full-workload F/T reverse weight mismatch")

    permutations = []
    raw_outcomes = []
    for repetition in range(args.public_repetitions):
        public_seed = matched.stable_seed(
            args.seed, deployment.get("dataset", "wp"), deployment_id, trial,
            "public", repetition
        )
        permutation = matched.public_permutation(len(tx_list), public_seed)
        permutations.append(permutation)
        raw_outcomes.append(matched.finalize_graph(
            tx_list, raw_adjacency, permutation=permutation
        ))

    deferred = int(attack["deferred_edge"])
    carryover_outcomes = apply_cross_epoch_admission(raw_outcomes, deferred)

    raw_final_rate = float(np.mean([
        outcome["final_ft"] for outcome in raw_outcomes
    ]))
    cross_epoch_final_rate = float(np.mean([
        outcome["final_ft"] for outcome in carryover_outcomes
    ]))

    case_file = ""
    if trial < args.conformance_trials_per_deployment:
        case_file = (
            f"{deployment.get('dataset', 'wp')}-d{deployment_id:02d}-"
            f"t{trial:03d}-plain-{report_behavior}.case"
        )
        matched.write_conformance_case(
            os.path.join(args.output_dir, "unspool-conformance", case_file),
            orderings, reporters, pair_order, permutations,
        )

    return {
        "unspool_transactions": len(tx_list),
        "unspool_public_repetitions": args.public_repetitions,
        "raw_direct_edge_FT": int(raw_adjacency[f_index, t_index]),
        "raw_direct_edge_TF": int(raw_adjacency[t_index, f_index]),
        "raw_final_F_before_T_rate": raw_final_rate,
        "raw_same_scc_rate": float(np.mean([
            outcome["same_scc"] for outcome in raw_outcomes
        ])),
        "cross_epoch_final_F_before_T_rate": cross_epoch_final_rate,
        "cross_epoch_T_before_F_rate": 1.0 - cross_epoch_final_rate,
        "cross_epoch_same_scc_rate": float(np.mean([
            outcome["same_scc"] for outcome in carryover_outcomes
        ])),
        "carried_F_to_next_epoch": deferred,
        "unspool_conformance_case": case_file,
    }


def profile_support_for_true_source(window, snapshot, reporter_locations,
                                    source, observed_gap, scale):
    """Construct a normal-profile witness for a generated benign pair.

    This is a workload-evaluation shortcut, not extra protocol evidence. The
    operational guard searches all candidate ingresses. For these generated
    controls, the ground-truth ingress must explain all 16 honest reports, or
    at least the 11 honest reports in the pair-swap sensitivity, so exhibiting
    this one witness is equivalent to establishing the existential profile
    predicate.
    """
    lower, upper = window.bounds(snapshot["bound_epoch"], scale)
    victim_index = window.index[snapshot["victim"]]
    source_index = window.index[source]
    reporter_indices = np.asarray(
        [
            window.index[matched.physical_location(value)]
            for value in reporter_locations
        ], dtype=int
    )
    interval_lower = (
        observed_gap
        + lower[victim_index, reporter_indices]
        - upper[source_index, reporter_indices]
    )
    interval_upper = (
        observed_gap
        + upper[victim_index, reporter_indices]
        - lower[source_index, reporter_indices]
    )
    support, _, delta = guard.best_interval_overlap(
        interval_lower[None, :], interval_upper[None, :]
    )
    return support, delta


def evaluate_background_behavior(window, snapshot, deployment, context,
                                 releases, sources, arrivals, behavior, args):
    reporters = deployment["reporter_order"][:16]
    reporter_locations = [deployment["replicas"][index] for index in reporters]
    byzantine = set(snapshot["corrupted"])
    byzantine_mask = np.asarray(
        [index in byzantine for index in reporters], dtype=bool
    )
    selected_t = snapshot["t_arrivals"][reporters]
    gaps = arrivals[:, reporters] - selected_t[None, :]
    gaps[np.abs(gaps) <= guard.EPS_MS] = 2 * guard.EPS_MS
    if behavior == "pair-swap":
        gaps[:, byzantine_mask] = -np.maximum(
            np.abs(gaps[:, byzantine_mask]), 2 * guard.EPS_MS
        )
    elif behavior != "honest":
        raise ValueError(behavior)

    w_ft = np.sum(gaps < -guard.EPS_MS, axis=1)
    w_tf = np.sum(gaps > guard.EPS_MS, axis=1)
    raw_edges = (w_ft >= 6) & (w_ft > w_tf)
    profile_support = np.empty(len(releases), dtype=np.int16)
    profile_delta = np.empty(len(releases), dtype=np.float64)
    for index, (source, gap) in enumerate(zip(sources, gaps)):
        profile_support[index], profile_delta[index] = (
            profile_support_for_true_source(
                window, snapshot, reporter_locations, source, gap,
                args.uncertainty_scale,
            )
        )
    profile_exists = profile_support >= args.profile_requirement

    feasible = np.zeros(len(releases), dtype=np.int8)
    candidate_count = np.zeros(len(releases), dtype=np.int32)
    query = np.flatnonzero(raw_edges & profile_exists)
    if len(query):
        result, counts = guard.trace_feasibility_batch(
            context, gaps[query], chunk_size=args.feasibility_chunk_size
        )
        feasible[query] = result
        candidate_count[query] = counts
    policy_deferred = {
        "no-guard": np.zeros(len(releases), dtype=bool),
        "profile-only": raw_edges & ~profile_exists,
        "capacity-only": raw_edges & bool(context["capacity_unsafe"]),
        "profile+trace-bound": (
            raw_edges & (~profile_exists | feasible.astype(bool))
        ),
        "carry-every-reverse-edge": raw_edges.copy(),
    }

    rows = []
    for index in range(len(releases)):
        row = {
            "behavior": behavior,
            "tx": f"B{index:03d}",
            "source": sources[index],
            "release_ms": float(releases[index]),
            "release_gap_ms": float(releases[index] - matched.T_GEN_TIME),
            "w_FT": int(w_ft[index]),
            "w_TF": int(w_tf[index]),
            "raw_strict_edge": int(raw_edges[index]),
            "profile_support": int(profile_support[index]),
            "profile_delta_ms": float(profile_delta[index]),
            "profile_exists": int(profile_exists[index]),
            "trace_attack_feasible": int(feasible[index]),
            "trace_feasible_candidate_count": int(candidate_count[index]),
            "false_deferred": int(
                policy_deferred["profile+trace-bound"][index]
            ),
        }
        for policy, values in policy_deferred.items():
            row[f"deferred_{policy_slug(policy)}"] = int(values[index])
        rows.append(row)
    return rows


def run_deployment(task):
    deployment_id, args_dict = task
    args = SimpleNamespace(**args_dict)
    trace = matched.LatencyTrace(args.ping_file, "half", 0.15)
    window = guard.WindowTrace(args.ping_file)
    deployment = matched.make_deployment(
        trace, deployment_id, guard.deployment_args(args)
    )
    pair_rows = []
    trial_rows = []
    for trial in range(args.trials_per_deployment):
        snapshot = guard.pair_snapshot(
            trace, window, deployment, deployment_id, trial, args
        )
        reporters = deployment["reporter_order"][:16]
        context = guard.corruption_capacity_context(
            window, snapshot["bound_epoch"], args.uncertainty_scale,
            snapshot["victim"], deployment["replicas"], reporters,
            args.rho_floor_ms,
        )
        attack_rows = guard.evaluate_pair(
            window, deployment, snapshot, "attack-fastest", "pair-swap",
            snapshot["attack_f_arrivals"], args.uncertainty_scale, args,
            context,
        )
        attack_by_policy = {row["policy"]: row for row in attack_rows}
        attack = attack_by_policy["profile+trace-bound"]
        if args.execution_model == "section4-matched":
            releases, sources, arrivals = generate_section4_background(
                trace, deployment, deployment_id, trial,
                args.background_txs, args.seed,
            )
        else:
            releases, sources, arrivals = generate_background(
                window, snapshot, deployment, deployment_id, trial,
                args.background_txs, args.seed,
            )
        unspool_by_policy = {
            policy: evaluate_attack_unspool(
                deployment, snapshot, arrivals, attack_by_policy[policy],
                deployment_id, trial, args,
            )
            for policy in COMPONENT_POLICIES
        }
        unspool = unspool_by_policy["profile+trace-bound"]
        common = {
            "dataset": deployment["dataset"],
            "deployment": deployment_id,
            "trial": trial,
            "victim": snapshot["victim"],
            "execution_snapshot_hour": snapshot["hour"],
        }
        trial_row = {
            **common,
            "attack_raw_edge": attack["raw_strict_edge"],
            "attack_w_FT": attack["w_FT"],
            "attack_w_TF": attack["w_TF"],
            "attack_deferred": attack["deferred_edge"],
            "attack_retained": attack["retained_edge"],
            **unspool,
        }
        for policy in COMPONENT_POLICIES:
            slug = policy_slug(policy)
            policy_attack = attack_by_policy[policy]
            policy_unspool = unspool_by_policy[policy]
            trial_row.update({
                f"attack_{slug}_retained": policy_attack["retained_edge"],
                f"attack_{slug}_deferred": policy_attack["deferred_edge"],
                f"attack_{slug}_final_F_before_T_rate": policy_unspool[
                    "cross_epoch_final_F_before_T_rate"
                ],
            })
        for behavior in ("honest", "pair-swap"):
            rows = evaluate_background_behavior(
                window, snapshot, deployment, context, releases, sources,
                arrivals, behavior, args,
            )
            for row in rows:
                row.update(common)
            pair_rows.extend(rows)
            raw = sum(row["raw_strict_edge"] for row in rows)
            deferred = sum(row["false_deferred"] for row in rows)
            prefix = behavior.replace("-", "_")
            trial_row.update({
                f"{prefix}_background_pairs": len(rows),
                f"{prefix}_raw_edges": raw,
                f"{prefix}_false_deferred": deferred,
                f"{prefix}_any_false_deferred": int(deferred > 0),
            })
            for policy in COMPONENT_POLICIES:
                slug = policy_slug(policy)
                policy_deferred = sum(
                    row[f"deferred_{slug}"] for row in rows
                )
                trial_row.update({
                    f"{prefix}_{slug}_deferred": policy_deferred,
                    f"{prefix}_{slug}_any_deferred": int(
                        policy_deferred > 0
                    ),
                })
        trial_rows.append(trial_row)
    return pair_rows, trial_rows


def bootstrap_metric(frame, numerator, denominator, seed, repetitions):
    return guard.bootstrap_ratio(
        frame, numerator, denominator, seed, repetitions
    )


def summarize(trials, pairs, args):
    rows = []
    for index, behavior in enumerate(("honest", "pair-swap")):
        prefix = behavior.replace("-", "_")
        numerator = f"{prefix}_false_deferred"
        all_pairs = f"{prefix}_background_pairs"
        raw_edges = f"{prefix}_raw_edges"
        for label, denominator in [
            ("all_background_pairs", all_pairs),
            ("raw_strict_edges", raw_edges),
        ]:
            den = float(trials[denominator].sum())
            value = float(trials[numerator].sum() / den) if den else math.nan
            low, high = bootstrap_metric(
                trials, numerator, denominator,
                args.seed + 1000 + index * 10 + len(rows),
                args.bootstrap_repetitions,
            )
            rows.append({
                "scenario": "benign-background",
                "behavior": behavior,
                "metric": f"false_deferred_over_{label}",
                "value": value,
                "ci95_low": low,
                "ci95_high": high,
                "numerator": int(trials[numerator].sum()),
                "denominator": int(den),
            })
        all_trials = trials.assign(all_trials=1)
        any_column = f"{prefix}_any_false_deferred"
        value = float(trials[any_column].mean())
        low, high = bootstrap_metric(
            all_trials, any_column, "all_trials",
            args.seed + 2000 + index, args.bootstrap_repetitions,
        )
        rows.append({
            "scenario": "benign-background",
            "behavior": behavior,
            "metric": "batch_has_any_false_deferred",
            "value": value,
            "ci95_low": low,
            "ci95_high": high,
            "numerator": int(trials[any_column].sum()),
            "denominator": len(trials),
        })

    attack_trials = trials.assign(all_trials=1)
    attack_metrics = [
        ("raw_strict_edge_over_all_trials", "attack_raw_edge", "all_trials"),
        ("detected_given_raw_edge", "attack_deferred", "attack_raw_edge"),
        ("retained_edge_over_all_trials", "attack_retained", "all_trials"),
    ]
    for metric_index, (metric, numerator, denominator) in enumerate(
        attack_metrics
    ):
        den = float(attack_trials[denominator].sum())
        value = float(attack_trials[numerator].sum() / den) if den else math.nan
        low, high = bootstrap_metric(
            attack_trials, numerator, denominator,
            args.seed + 3000 + metric_index, args.bootstrap_repetitions,
        )
        rows.append({
            "scenario": "attack-fastest",
            "behavior": "pair-swap",
            "metric": metric,
            "value": value,
            "ci95_low": low,
            "ci95_high": high,
            "numerator": int(attack_trials[numerator].sum()),
            "denominator": int(den),
        })

    final_metrics = [
        "raw_final_F_before_T_rate",
        "raw_same_scc_rate",
        "cross_epoch_final_F_before_T_rate",
        "cross_epoch_T_before_F_rate",
        "cross_epoch_same_scc_rate",
    ]
    for metric_index, metric in enumerate(final_metrics):
        numerator = f"summary_{metric}"
        final_trials = trials.assign(
            all_trials=1,
            **{numerator: trials[metric]},
        )
        value = float(final_trials[metric].mean())
        low, high = bootstrap_metric(
            final_trials, numerator, "all_trials",
            args.seed + 4000 + metric_index, args.bootstrap_repetitions,
        )
        rows.append({
            "scenario": "attack-fastest",
            "behavior": "pair-swap",
            "metric": metric,
            "value": value,
            "ci95_low": low,
            "ci95_high": high,
            "numerator": float(final_trials[numerator].sum()),
            "denominator": len(final_trials),
        })

    pair_frame = pairs.copy()
    pair_frame["release_bin"] = pd.cut(
        pair_frame.release_gap_ms,
        [-math.inf, -50, -20, -10, 0, 10, 20, 50, math.inf],
        right=False,
    ).astype(str)
    bins = pair_frame.groupby(
        ["behavior", "release_bin"], observed=True
    ).agg(
        pairs=("tx", "size"),
        raw_edges=("raw_strict_edge", "sum"),
        false_deferred=("false_deferred", "sum"),
    ).reset_index()
    bins["false_positive_rate"] = bins.false_deferred / bins.pairs
    bins["conditional_false_positive_rate"] = (
        bins.false_deferred / bins.raw_edges.replace(0, np.nan)
    )
    return pd.DataFrame(rows), bins


def summarize_components(trials, args):
    rows = []
    frame = trials.assign(all_trials=1)
    for policy_index, policy in enumerate(COMPONENT_POLICIES):
        slug = policy_slug(policy)
        specs = [
            (
                "attack_edge_retained_given_raw_edge",
                f"attack_{slug}_retained", "attack_raw_edge",
            ),
            (
                "guarded_final_F_before_T",
                f"attack_{slug}_final_F_before_T_rate", "all_trials",
            ),
            (
                "benign_pair_carried",
                f"honest_{slug}_deferred", "honest_background_pairs",
            ),
            (
                "batch_any_benign_carry",
                f"honest_{slug}_any_deferred", "all_trials",
            ),
        ]
        for metric_index, (metric, numerator, denominator) in enumerate(specs):
            den = float(frame[denominator].sum())
            value = float(frame[numerator].sum() / den) if den else math.nan
            low, high = bootstrap_metric(
                frame, numerator, denominator,
                args.seed + 8000 + policy_index * 100 + metric_index,
                args.bootstrap_repetitions,
            )
            rows.append({
                "policy": policy,
                "metric": metric,
                "value": value,
                "ci95_low": low,
                "ci95_high": high,
                "numerator": float(frame[numerator].sum()),
                "denominator": int(den),
                "deployments": int(frame.deployment.nunique()),
                "trials": len(frame),
            })
    return pd.DataFrame(rows)


def write_report(summary, bins, output_dir, args):
    def row(metric, behavior):
        return summary[
            (summary.metric == metric) & (summary.behavior == behavior)
        ].iloc[0]

    lines = [
        "# Corruption-Aware Guard with Poisson Background",
        "",
        f"Each trial contains `{args.background_txs}` independent benign "
        "transactions from 15 clients. Each client independently generates "
        "its assigned transactions with cumulative exponential gaps whose "
        "mean is `1000/client_tx_count` ms, exactly matching the original "
        "artifact's workload generator. The guard checks every background "
        "transaction against T; it does not check all background/background pairs.",
        "",
        "| Reports | Pair false positive | Conditional on raw edge | Batch has any false positive |",
        "|---|---:|---:|---:|",
    ]
    for behavior in ("honest", "pair-swap"):
        all_pairs = row("false_deferred_over_all_background_pairs", behavior)
        conditional = row("false_deferred_over_raw_strict_edges", behavior)
        batch = row("batch_has_any_false_deferred", behavior)
        lines.append(
            f"| {behavior} | {100 * all_pairs.value:.2f}% "
            f"[{100 * all_pairs.ci95_low:.2f}, {100 * all_pairs.ci95_high:.2f}] | "
            f"{100 * conditional.value:.2f}% "
            f"[{100 * conditional.ci95_low:.2f}, {100 * conditional.ci95_high:.2f}] | "
            f"{100 * batch.value:.1f}% "
            f"[{100 * batch.ci95_low:.1f}, {100 * batch.ci95_high:.1f}] |"
        )
    raw_attack = row("raw_strict_edge_over_all_trials", "pair-swap")
    attack = row("detected_given_raw_edge", "pair-swap")
    retained = row("retained_edge_over_all_trials", "pair-swap")
    raw_final = row("raw_final_F_before_T_rate", "pair-swap")
    cross_final = row("cross_epoch_final_F_before_T_rate", "pair-swap")
    lines.extend([
        "",
        f"The fastest attack creates a raw strict edge in "
        f"**{100 * raw_attack.value:.1f}%** of trials "
        f"[{100 * raw_attack.ci95_low:.1f}, {100 * raw_attack.ci95_high:.1f}] "
        f"({int(raw_attack.numerator)}/{int(raw_attack.denominator)}). "
        f"Detection conditional on that edge is "
        f"**{100 * attack.value:.1f}%** "
        f"[{100 * attack.ci95_low:.1f}, {100 * attack.ci95_high:.1f}] "
        f"({int(attack.numerator)}/{int(attack.denominator)}), leaving a "
        f"retained attack edge in **{100 * retained.value:.1f}%** of all trials "
        f"({int(retained.numerator)}/{int(retained.denominator)}).",
        "",
        f"The complete 302-transaction Themis graph and Unspool finalizer put "
        f"F before T in **{100 * raw_final.value:.1f}%** of raw executions. "
        f"Admission-only carryover of detected F transactions to the post-T "
        f"epoch leaves "
        f"**{100 * cross_final.value:.1f}%** final F-before-T outcomes across "
        f"all trials; these residual outcomes come only from trials where the "
        f"guard did not trigger.",
        "No synthetic same-epoch T-to-F edge is constructed or evaluated.",
        "",
        "Release-gap buckets are stored in `release_gap_summary.csv`. A batch-"
        "level false positive can be high even when pair-level selectivity is "
        "good, because 300 hypotheses are tested in every trial.",
    ])
    with open(os.path.join(output_dir, "RESULTS.md"), "w") as file:
        file.write("\n".join(lines) + "\n")


def run(args):
    if not 1 <= args.workers <= 5:
        raise ValueError("workers must be between 1 and 5")
    os.makedirs(args.output_dir, exist_ok=True)
    tasks = [(value, vars(args)) for value in range(args.deployments)]
    pair_rows = []
    trial_rows = []
    if args.workers == 1:
        iterator = map(run_deployment, tasks)
        for pairs, trials in tqdm(
            iterator, total=len(tasks), desc="background deployments"
        ):
            pair_rows.extend(pairs)
            trial_rows.extend(trials)
    else:
        with Pool(args.workers) as pool:
            for pairs, trials in tqdm(
                pool.imap(run_deployment, tasks), total=len(tasks),
                desc="background deployments",
            ):
                pair_rows.extend(pairs)
                trial_rows.extend(trials)
    pairs = pd.DataFrame(pair_rows)
    trials = pd.DataFrame(trial_rows)
    pairs.to_csv(
        os.path.join(args.output_dir, "background_pair_results.csv.gz"),
        index=False, compression="gzip",
    )
    trials.to_csv(
        os.path.join(args.output_dir, "trial_summary.csv"), index=False
    )
    summary, bins = summarize(trials, pairs, args)
    component_summary = summarize_components(trials, args)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    component_summary.to_csv(
        os.path.join(args.output_dir, "component_summary.csv"), index=False
    )
    bins.to_csv(
        os.path.join(args.output_dir, "release_gap_summary.csv"), index=False
    )
    baseline_args = guard.deployment_args(args)
    baseline_trace = matched.LatencyTrace(args.ping_file, "half", 0.15)
    expected_section4_layout = (
        "global-random"
        if baseline_trace.dataset == "aws"
        else "latency-clustered"
    )
    baseline_deployments = []
    for deployment_id in range(args.deployments):
        deployment = matched.make_deployment(
            baseline_trace, deployment_id, baseline_args
        )
        baseline_deployments.append({
            "deployment": deployment_id,
            "placement_candidate_count": deployment[
                "corrupted_placement_info"
            ]["candidate_count"],
            "corrupted_replica_ids": deployment["corrupted_replica_indices"],
            "corrupted_locations": [
                matched.physical_location(deployment["replicas"][index])
                for index in deployment["corrupted_replica_indices"]
            ],
        })
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump({
            "arguments": vars(args),
            "attack_baseline": {
                "name": (
                    "section4-matched-five-corrupted-replica-joint"
                    if args.execution_model == "section4-matched"
                    else "constrained-five-corrupted-replica-local-union"
                ),
                "same_as_section4_global_joint": bool(
                    args.execution_model == "section4-matched"
                    and args.allow_corrupted_region_colocation
                    and args.replica_layout == expected_section4_layout
                    and args.background_txs == 300
                ),
                "num_nodes": 21,
                "faults": 5,
                "replica_layout": args.replica_layout,
                "replica_regions": args.replica_regions,
                "execution_model": args.execution_model,
                "observation_model": "corrupted-replica-direct",
                "trigger": (
                    "local: each corrupted replica releases after its own T "
                    "arrival; release times need not be equal"
                ),
                "rho_ms": args.rho_ms,
                "corrupted_placement": "history-optimized-global",
                "allow_corrupted_region_colocation": (
                    args.allow_corrupted_region_colocation
                ),
                "placement_candidate_policy": (
                    "all measured trace locations"
                    if args.allow_corrupted_region_colocation
                    else "trace locations not occupied by an honest replica"
                ),
                "submission_api_or_relay_count": 0,
                "byzantine_reports": 5,
                "byzantine_report_mode": "pair-swap",
                "reporter_selection": (
                    "first 16 identities in the frozen reporter order, "
                    "including all five corrupted replicas"
                ),
                "dissemination": "plain first-copy",
                "deployments": baseline_deployments,
            },
            "workload": (
                "Section IV matched workload: 300 transactions use the same "
                "shuffled client assignments, cumulative exponential arrivals, "
                "execution samples, and transaction identifiers"
                if args.execution_model == "section4-matched"
                else "300 benign transactions split evenly across 15 clients; "
                "each client uses cumulative exponential gaps with mean "
                "1000/client_tx_count ms"
            ),
            "pair_scope": "all 300 background transactions paired with T",
            "benign_profile_witness": (
                "the simulator evaluates the generated transaction's true "
                "ingress as a constructive witness for the operational "
                "exists-an-ingress predicate; it is not exposed to the guard"
            ),
            "finalizer": (
                "complete Python port of Themis weighting, graph construction, "
                "SCC decomposition, and Hamiltonian Unspool; representative "
                "raw cases are emitted for instrumented artifact C++ conformance"
            ),
            "deferral_semantics": (
                "detected F is excluded from the current epoch and becomes "
                "eligible only after T finalizes; it is not deleted"
            ),
            "same_epoch_edge_reversal": "not evaluated",
            "component_policies": list(COMPONENT_POLICIES),
            "limitation": (
                "does not evaluate all background/background pairs or semantic "
                "dependency filtering"
                + (
                    ""
                    if args.execution_model == "section4-matched"
                    else "; the attack baseline is not the Section 4 global-joint "
                    "placement"
                )
            ),
        }, file, indent=2)
    write_report(summary, bins, args.output_dir, args)


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
            "corruption-aware-guard-background00"
        ),
    )
    parser.add_argument("--deployments", type=int, default=20)
    parser.add_argument("--trials_per_deployment", type=int, default=20)
    parser.add_argument("--background_txs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--replica_layout",
        choices=["global-random", "latency-clustered"],
        default="latency-clustered",
    )
    parser.add_argument("--replica_regions", type=int, default=5)
    parser.add_argument(
        "--allow_corrupted_region_colocation", action="store_true"
    )
    parser.add_argument(
        "--execution_model",
        choices=["guard-window", "section4-matched"],
        default="guard-window",
    )
    parser.add_argument("--rho_ms", type=float, default=0.0)
    parser.add_argument("--rho_floor_ms", type=float, default=0.0)
    parser.add_argument(
        "--bound_mode", choices=["same-hour", "previous-hour"],
        default="same-hour",
    )
    parser.add_argument("--profile_requirement", type=int, default=11)
    parser.add_argument("--uncertainty_scale", type=float, default=1.0)
    parser.add_argument("--feasibility_chunk_size", type=int, default=16)
    parser.add_argument("--public_repetitions", type=int, default=20)
    parser.add_argument(
        "--conformance_trials_per_deployment", type=int, default=1
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
