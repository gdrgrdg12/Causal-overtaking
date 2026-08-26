#!/usr/bin/env python3
"""Evaluate a corruption-aware latency guard on the global-direct baseline."""

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

import matched_multientry_evaluation as matched


EPS_MS = 1e-9


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


class WindowTrace:
    def __init__(self, path):
        self.sample_values = None
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                self.locations = archive["locations"].astype(str).tolist()
                self.hours = archive["hours"].astype(str).tolist()
                self.minimum = archive["min_owd"].astype(np.float64)
                self.average = archive["avg_owd"].astype(np.float64)
                self.maximum = archive["max_owd"].astype(np.float64)
                self.mdev = archive["mdev_owd"].astype(np.float64)
                self.metadata = json.loads(str(archive["metadata"]))
        else:
            self._init_sample_partitions(path)
        self.index = {
            location: position for position, location in enumerate(self.locations)
        }

    def _init_sample_partitions(self, path):
        frame = pd.read_csv(path)
        value_column = "time" if "time" in frame.columns else "avg"
        frame = frame[["source", "destination", value_column]].dropna()
        frame = frame[frame[value_column] >= 0]
        self.locations = sorted(frame.source.astype(str).unique().tolist())
        self.hours = ["even-sample history", "odd-sample execution"]
        size = len(self.locations)
        index = {location: offset for offset, location in enumerate(self.locations)}
        shape = (2, size, size)
        self.minimum = np.zeros(shape, dtype=np.float64)
        self.average = np.zeros(shape, dtype=np.float64)
        self.maximum = np.zeros(shape, dtype=np.float64)
        self.mdev = np.zeros(shape, dtype=np.float64)
        self.sample_values = {}
        for (source, destination), group in frame.groupby(
            ["source", "destination"], sort=False
        ):
            values = group[value_column].to_numpy(dtype=np.float64) / 2.0
            partitions = [values[::2], values[1::2]]
            if not len(partitions[1]):
                partitions[1] = partitions[0]
            source = str(source)
            destination = str(destination)
            row = index[source]
            column = index[destination]
            for epoch, partition in enumerate(partitions):
                if not len(partition):
                    raise ValueError(
                        f"missing latency samples for {source}->{destination}"
                    )
                self.minimum[epoch, row, column] = float(np.min(partition))
                self.average[epoch, row, column] = float(np.mean(partition))
                self.maximum[epoch, row, column] = float(np.max(partition))
                self.mdev[epoch, row, column] = float(np.std(partition))
                self.sample_values[(epoch, source, destination)] = partition
        self.metadata = {
            "dataset": "aws" if value_column == "time" else "wp",
            "source": path,
            "delay_model": (
                "even-indexed RTT/2 samples for frozen history and "
                "odd-indexed RTT/2 samples for execution"
            ),
        }

    def bounds(self, epoch, scale):
        center = self.average[epoch]
        lower = np.maximum(
            0.0, center - scale * (center - self.minimum[epoch])
        )
        upper = center + scale * (self.maximum[epoch] - center)
        return lower, upper

    def sample(self, epoch, source, destination, seed, *parts):
        """Match the snapshot-window execution model used by yesterday's run."""
        source = matched.physical_location(source)
        destination = matched.physical_location(destination)
        if getattr(self, "sample_values", None) is not None:
            values = self.sample_values[(epoch, source, destination)]
            rng = matched.stable_rng(
                seed, "sample-window", epoch, source, destination, *parts
            )
            return float(values[rng.randrange(len(values))])
        source_index = self.index[str(source)]
        destination_index = self.index[str(destination)]
        center = self.average[epoch, source_index, destination_index]
        sigma = self.mdev[epoch, source_index, destination_index]
        lower = self.minimum[epoch, source_index, destination_index]
        upper = self.maximum[epoch, source_index, destination_index]
        rng = matched.stable_rng(
            seed, "wp-snapshot-window", epoch, source, destination, *parts
        )
        for _ in range(64):
            value = center + rng.gauss(0.0, sigma)
            if lower <= value <= upper:
                return float(value)
        return float(min(upper, max(lower, center)))


def deployment_args(args):
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
        placement_victim_scope="all-locations",
        allow_corrupted_region_colocation=getattr(
            args, "allow_corrupted_region_colocation", False
        ),
        byzantine_reports=5,
        training_snapshots=24,
        relay_entry_count=220,
        rho_ms=args.rho_ms,
    )


def strict_edge(report_t, report_f, threshold=6):
    forward = int(np.sum(report_f < report_t - EPS_MS))
    reverse = int(np.sum(report_t < report_f - EPS_MS))
    return forward, reverse, int(
        forward >= threshold and forward > reverse
    )


def encode_t_before_f_ties(t_arrivals, f_arrivals):
    """Mirror the full ordering's stable T-before-F rule for time ties."""
    report_t = np.asarray(t_arrivals, dtype=np.float64)
    report_f = np.asarray(f_arrivals, dtype=np.float64).copy()
    ties = np.abs(report_t - report_f) <= EPS_MS
    report_f[ties] = report_t[ties] + 2 * EPS_MS
    return report_f


def apply_report_behavior(t_arrivals, f_arrivals, byzantine_mask, behavior):
    report_t = np.asarray(t_arrivals, dtype=np.float64).copy()
    report_f = np.asarray(f_arrivals, dtype=np.float64).copy()
    if behavior == "honest":
        return report_t, report_f
    if behavior != "pair-swap":
        raise ValueError(behavior)
    swap = byzantine_mask & (report_t < report_f)
    original_t = report_t[swap].copy()
    report_t[swap] = report_f[swap]
    report_f[swap] = original_t
    return report_t, report_f


def best_interval_overlap(lower, upper):
    """Maximum overlap for one source row; never mixes source candidates."""
    candidates = np.concatenate((lower, upper), axis=1)
    support = (
        (lower[:, None, :] <= candidates[:, :, None] + EPS_MS)
        & (upper[:, None, :] >= candidates[:, :, None] - EPS_MS)
    ).sum(axis=2)
    flat = int(np.argmax(support))
    source, point = np.unravel_index(flat, support.shape)
    return int(support[source, point]), int(source), float(
        candidates[source, point]
    )


def profile_fit(window, epoch, scale, victim, reporters, observed_gap):
    lower, upper = window.bounds(epoch, scale)
    victim_index = window.index[victim]
    reporter_indices = np.asarray(
        [
            window.index[matched.physical_location(reporter)]
            for reporter in reporters
        ], dtype=int
    )
    t_lower = lower[victim_index, reporter_indices]
    t_upper = upper[victim_index, reporter_indices]
    f_lower = lower[:, reporter_indices]
    f_upper = upper[:, reporter_indices]
    interval_lower = observed_gap[None, :] + t_lower[None, :] - f_upper
    interval_upper = observed_gap[None, :] + t_upper[None, :] - f_lower
    support, source_index, delta = best_interval_overlap(
        interval_lower, interval_upper
    )
    return {
        "profile_support": support,
        "profile_source": window.locations[source_index],
        "profile_delta_ms": delta,
    }


def trace_feasible_subsets(source_earliest_gap, reporters, observed_gap,
                           subset_size):
    """Find corruption sets that can reproduce the current report vector.

    Corrupted reporters may report arbitrary gaps. Every other reporter must
    receive F no earlier than its same-corrupt-source causal floor; arbitrary
    non-negative per-connection pacing is allowed after that floor.
    """
    source_earliest_gap = np.asarray(source_earliest_gap, dtype=np.float64)
    reporters = np.asarray(reporters, dtype=int)
    observed_gap = np.asarray(observed_gap, dtype=np.float64)
    subsets = np.asarray(list(itertools.combinations(
        range(source_earliest_gap.shape[0]), subset_size
    )), dtype=int)
    subset_earliest = np.min(source_earliest_gap[subsets], axis=1)
    corrupt_reporters = (
        subsets[:, :, None] == reporters[None, None, :]
    ).any(axis=1)
    physically_possible = (
        observed_gap[None, :] >= subset_earliest - EPS_MS
    )
    feasible = np.all(physically_possible | corrupt_reporters, axis=1)
    return subsets, subset_earliest, corrupt_reporters, feasible


def corruption_capacity_context(window, epoch, scale, victim, replicas,
                                reporters, rho_floor_ms):
    """Precompute all same-source five-corruption path lower bounds."""
    lower, upper = window.bounds(epoch, scale)
    replica_locations = np.asarray(
        [
            window.index[matched.physical_location(replica)]
            for replica in replicas
        ], dtype=int
    )
    reporter_location_indices = np.asarray(
        [
            window.index[matched.physical_location(replicas[index])]
            for index in reporters
        ], dtype=int
    )
    victim_index = window.index[victim]
    t_latest = upper[victim_index, reporter_location_indices]

    source_earliest_gap = np.empty(
        (len(replicas), len(reporters)), dtype=np.float64
    )
    for replica_index, location_index in enumerate(replica_locations):
        earliest = (
            lower[victim_index, location_index]
            + rho_floor_ms
            + lower[location_index, reporter_location_indices]
        )
        source_earliest_gap[replica_index] = earliest - t_latest

    subsets = np.asarray(list(itertools.combinations(
        range(source_earliest_gap.shape[0]), 5
    )), dtype=int)
    subset_earliest = np.min(source_earliest_gap[subsets], axis=1)
    corrupt_reporters = (
        subsets[:, :, None] == np.asarray(reporters)[None, None, :]
    ).any(axis=1)
    possible_winners = subset_earliest < -EPS_MS
    honest_support = np.sum(
        possible_winners & ~corrupt_reporters, axis=1
    )
    byzantine_support = np.sum(corrupt_reporters, axis=1)
    total_support = honest_support + byzantine_support
    margin = 2 * total_support - len(reporters)
    strict = (total_support >= 6) & (margin > 0)
    order = np.lexsort((honest_support, total_support, margin, strict))
    best_index = int(order[-1])
    return {
        "subsets": subsets,
        "subset_earliest": subset_earliest,
        "corrupt_reporters": corrupt_reporters,
        "strict": strict,
        "capacity_unsafe": int(np.any(strict)),
        "capacity_max_honest": int(honest_support[best_index]),
        "capacity_max_total": int(total_support[best_index]),
        "capacity_max_margin": int(margin[best_index]),
        "capacity_witness": ";".join(
            str(value) for value in subsets[best_index]
        ),
    }


def trace_feasibility(capacity_context, observed_gap):
    physically_possible = (
        np.asarray(observed_gap)[None, :]
        >= capacity_context["subset_earliest"] - EPS_MS
    )
    feasible = np.all(
        physically_possible | capacity_context["corrupt_reporters"], axis=1
    )
    trace_indices = np.flatnonzero(feasible & capacity_context["strict"])
    trace_index = int(trace_indices[0]) if len(trace_indices) else -1
    subsets = capacity_context["subsets"]
    return {
        "trace_attack_feasible": int(trace_index >= 0),
        "trace_feasible_candidate_count": int(np.sum(
            feasible & capacity_context["strict"]
        )),
        "trace_feasible_witness": (
            ";".join(str(value) for value in subsets[trace_index])
            if trace_index >= 0 else ""
        ),
    }


def trace_feasibility_batch(capacity_context, observed_gaps, chunk_size=32):
    """Vectorized current-trace feasibility for multiple report vectors."""
    observed_gaps = np.asarray(observed_gaps, dtype=np.float64)
    if observed_gaps.ndim != 2:
        raise ValueError("observed_gaps must have shape pair x report")
    floors = capacity_context["subset_earliest"]
    ignored = capacity_context["corrupt_reporters"]
    strict = capacity_context["strict"]
    feasible_result = np.zeros(len(observed_gaps), dtype=np.int8)
    candidate_counts = np.zeros(len(observed_gaps), dtype=np.int32)
    for start in range(0, len(observed_gaps), chunk_size):
        stop = min(len(observed_gaps), start + chunk_size)
        possible = (
            observed_gaps[start:stop, None, :] >= floors[None, :, :] - EPS_MS
        ) | ignored[None, :, :]
        feasible = np.all(possible, axis=2) & strict[None, :]
        feasible_result[start:stop] = np.any(feasible, axis=1)
        candidate_counts[start:stop] = np.sum(feasible, axis=1)
    return feasible_result, candidate_counts


def corruption_capacity_bound(capacity_context, observed_gap):
    """Combine precomputed global capacity with current-trace feasibility."""
    trace = trace_feasibility(capacity_context, observed_gap)
    return {
        key: value for key, value in capacity_context.items()
        if key not in {
            "subsets", "subset_earliest", "corrupt_reporters", "strict"
        }
    } | trace


def bound_epoch_for_execution(execution_epoch, mode):
    if mode == "same-hour":
        return execution_epoch
    if mode == "previous-hour":
        if execution_epoch <= 0:
            raise ValueError("previous-hour bound is unavailable")
        return execution_epoch - 1
    raise ValueError(mode)


def actual_corruption_set_covered(capacity_context, observed_gap, corrupted):
    target = np.asarray(sorted(corrupted), dtype=int)
    matches = np.flatnonzero(np.all(
        capacity_context["subsets"] == target[None, :], axis=1
    ))
    if len(matches) != 1:
        raise AssertionError("actual corruption set missing from enumeration")
    index = int(matches[0])
    possible = (
        np.asarray(observed_gap) >=
        capacity_context["subset_earliest"][index] - EPS_MS
    ) | capacity_context["corrupt_reporters"][index]
    return int(np.all(possible))


def pair_snapshot(trace, window, deployment, deployment_id, trial, args):
    if trace.is_hourly_snapshot:
        trace.set_snapshot_context(
            "execution", args.seed, "trial", deployment_id, trial
        )
        epoch = trace.active_snapshot["execution"]
    else:
        epoch = 1
    bound_mode = getattr(args, "bound_mode", "same-hour")
    bound_epoch = bound_epoch_for_execution(epoch, bound_mode)
    replicas = deployment["replicas"]
    victim = matched.stable_rng(
        args.seed, trace.dataset, deployment_id, trial, "victim"
    ).choice(deployment["clients"])
    section4_matched = getattr(args, "execution_model", "guard-window") == (
        "section4-matched"
    )
    if section4_matched:
        t_arrivals = matched.T_GEN_TIME + np.asarray([
            trace.sample(
                "execution", victim, replica, args.seed,
                "trial", deployment_id, trial, "T", replica,
            )
            for replica in replicas
        ])
    else:
        t_arrivals = matched.T_GEN_TIME + np.asarray([
            window.sample(
                epoch, victim, replica, args.seed,
                deployment_id, trial, "T", replica,
            )
            for replica in replicas
        ])

    corrupted = matched.corrupted_replica_indices(
        deployment, deployment_args(args)
    )
    observations = t_arrivals[np.asarray(corrupted, dtype=int)]
    direct = np.empty((len(corrupted), len(replicas)), dtype=np.float64)
    if section4_matched:
        entries = deployment["entries"]
        if len(entries) != len(corrupted):
            raise AssertionError("corrupted observer/entry count mismatch")
        for source_position, (corrupted_index, entry) in enumerate(
            zip(corrupted, entries)
        ):
            if entry.get("source_replica_index") != corrupted_index:
                raise AssertionError("corrupted observer/entry identity mismatch")
            for replica_index, replica in enumerate(replicas):
                direct[source_position, replica_index] = (
                    observations[source_position]
                    + args.rho_ms
                    + trace.sample(
                        "execution", entry["location"], replica, args.seed,
                        "trial", deployment_id, trial,
                        "submit", entry["id"], replica,
                    )
                )
    else:
        for source_position, corrupted_index in enumerate(corrupted):
            source = replicas[corrupted_index]
            for replica_index, replica in enumerate(replicas):
                direct[source_position, replica_index] = (
                    observations[source_position]
                    + args.rho_ms
                    + window.sample(
                        epoch, source, replica, args.seed,
                        deployment_id, trial, "attack-F", source, replica,
                    )
                )
    attack_f = np.min(direct, axis=0)

    benign_candidates = [
        source for source in deployment["clients"] if source != victim
    ]
    benign_source = matched.stable_rng(
        args.seed, trace.dataset, deployment_id, trial, "guard-benign-source"
    ).choice(benign_candidates)
    if section4_matched:
        benign_base = matched.T_GEN_TIME + np.asarray([
            trace.sample(
                "execution", benign_source, replica, args.seed,
                "trial", deployment_id, trial,
                "guard-benign-F", benign_source, replica,
            )
            for replica in replicas
        ])
    else:
        benign_base = matched.T_GEN_TIME + np.asarray([
            window.sample(
                epoch, benign_source, replica, args.seed,
                deployment_id, trial, "benign-F", benign_source, replica,
            )
            for replica in replicas
        ])
    return {
        "epoch": epoch,
        "hour": window.hours[epoch],
        "bound_mode": bound_mode,
        "bound_epoch": bound_epoch,
        "bound_hour": window.hours[bound_epoch],
        "victim": victim,
        "benign_source": benign_source,
        "t_arrivals": t_arrivals,
        "attack_f_arrivals": attack_f,
        "benign_f_base": benign_base,
        "corrupted": corrupted,
    }


def evaluate_pair(window, deployment, snapshot, scenario, report_behavior,
                  f_arrivals, scale, args, capacity_context=None):
    reporters = deployment["reporter_order"][:16]
    reporter_locations = [deployment["replicas"][index] for index in reporters]
    byzantine = set(snapshot["corrupted"])
    byzantine_mask = np.asarray(
        [index in byzantine for index in reporters], dtype=bool
    )
    selected_t = snapshot["t_arrivals"][reporters]
    selected_f = encode_t_before_f_ties(
        selected_t, np.asarray(f_arrivals, dtype=np.float64)[reporters]
    )
    report_t, report_f = apply_report_behavior(
        selected_t, selected_f, byzantine_mask, report_behavior
    )
    w_ft, w_tf, raw_edge = strict_edge(report_t, report_f)
    profile = profile_fit(
        window, snapshot["bound_epoch"], scale, snapshot["victim"],
        reporter_locations, report_f - report_t,
    )
    if capacity_context is None:
        capacity_context = corruption_capacity_context(
            window, snapshot["bound_epoch"], scale, snapshot["victim"],
            deployment["replicas"], reporters, args.rho_floor_ms,
        )
    capacity = corruption_capacity_bound(
        capacity_context, report_f - report_t
    )
    actual_honest_support = int(np.sum(
        (selected_f < selected_t - EPS_MS) & ~byzantine_mask
    ))
    base = {
        "scenario": scenario,
        "report_behavior": report_behavior,
        "uncertainty_scale": scale,
        "raw_strict_edge": raw_edge,
        "w_FT": w_ft,
        "w_TF": w_tf,
        "actual_honest_support": actual_honest_support,
        "actual_corruption_set_covered": (
            actual_corruption_set_covered(
                capacity_context, report_f - report_t, snapshot["corrupted"]
            ) if scenario == "attack-fastest" else math.nan
        ),
        "profile_requirement": args.profile_requirement,
        "profile_exists": int(
            profile["profile_support"] >= args.profile_requirement
        ),
        **profile,
        **capacity,
    }
    rows = []
    for policy in (
        "no-guard", "profile-only", "capacity-only",
        "profile+trace-bound", "profile+global-capacity",
        "carry-every-reverse-edge",
    ):
        accepted = base["profile_exists"]
        if policy == "no-guard":
            accepted = True
        elif policy == "capacity-only":
            accepted = not capacity["capacity_unsafe"]
        elif policy == "profile+trace-bound":
            accepted = accepted and not capacity["trace_attack_feasible"]
        elif policy == "profile+global-capacity":
            accepted = accepted and not capacity["capacity_unsafe"]
        elif policy == "carry-every-reverse-edge":
            accepted = False
        retained = int(raw_edge and accepted)
        rows.append({
            **base,
            "policy": policy,
            "retained_edge": retained,
            "deferred_edge": raw_edge - retained,
            "defer_no_profile": int(raw_edge and not base["profile_exists"]),
            "defer_unsafe_capacity": int(
                raw_edge and base["profile_exists"]
                and policy in {"capacity-only", "profile+global-capacity"}
                and capacity["capacity_unsafe"]
            ),
            "defer_trace_feasible": int(
                raw_edge and base["profile_exists"]
                and policy == "profile+trace-bound"
                and capacity["trace_attack_feasible"]
            ),
        })
    return rows


def run_deployment(task):
    deployment_id, args_dict = task
    args = SimpleNamespace(**args_dict)
    trace = matched.LatencyTrace(args.ping_file, "half", 0.15)
    window = WindowTrace(args.ping_file)
    deployment = matched.make_deployment(
        trace, deployment_id, deployment_args(args)
    )
    rows = []
    for trial in range(args.trials_per_deployment):
        snapshot = pair_snapshot(
            trace, window, deployment, deployment_id, trial, args
        )
        for scale in args.uncertainty_scales:
            capacity_context = corruption_capacity_context(
                window, snapshot["bound_epoch"], scale, snapshot["victim"],
                deployment["replicas"],
                deployment["reporter_order"][:16], args.rho_floor_ms,
            )
            attack_rows = evaluate_pair(
                window, deployment, snapshot, "attack-fastest", "pair-swap",
                snapshot["attack_f_arrivals"], scale, args, capacity_context,
            )
            for row in attack_rows:
                row["benign_delta_ms"] = math.nan
            rows.extend(attack_rows)
            for delta in args.benign_deltas_ms:
                benign_f = snapshot["benign_f_base"] + delta
                for behavior in ("honest", "pair-swap"):
                    benign_rows = evaluate_pair(
                        window, deployment, snapshot, "benign-single-ingress",
                        behavior, benign_f, scale, args, capacity_context,
                    )
                    for row in benign_rows:
                        row["benign_delta_ms"] = delta
                    rows.extend(benign_rows)
        for row in rows[-(
            len(args.uncertainty_scales)
            * (3 + len(args.benign_deltas_ms) * 6)
        ):]:
            row.update({
                "dataset": "wp",
                "deployment": deployment_id,
                "trial": trial,
                "victim": snapshot["victim"],
                "execution_snapshot_hour": snapshot["hour"],
                "bound_mode": snapshot["bound_mode"],
                "bound_snapshot_hour": snapshot["bound_hour"],
                "benign_source": snapshot["benign_source"],
                "corrupted_locations": ";".join(
                    deployment["replicas"][index]
                    for index in snapshot["corrupted"]
                ),
            })
    return rows


def bootstrap_ratio(group, numerator, denominator, seed, repetitions):
    deployments = sorted(group.deployment.unique())
    rng = np.random.default_rng(seed)
    locals_ = [group[group.deployment == value] for value in deployments]
    lengths = {len(local) for local in locals_}
    values = []
    if len(lengths) == 1:
        width = lengths.pop()
        num = np.stack(
            [local[numerator].to_numpy(dtype=float) for local in locals_]
        )
        den = np.stack(
            [local[denominator].to_numpy(dtype=float) for local in locals_]
        )
        count = len(deployments)
        for start in range(0, repetitions, 250):
            batch = min(250, repetitions - start)
            selected = rng.integers(0, count, size=(batch, count))
            sampled = rng.integers(
                0, width, size=(batch, count, width)
            )
            chosen_num = num[selected[:, :, None], sampled].sum(axis=(1, 2))
            chosen_den = den[selected[:, :, None], sampled].sum(axis=(1, 2))
            batch_values = np.divide(
                chosen_num,
                chosen_den,
                out=np.full(batch, np.nan),
                where=chosen_den > 0,
            )
            values.extend(batch_values.tolist())
    else:
        for _ in range(repetitions):
            selected = rng.integers(
                0, len(deployments), len(deployments)
            )
            total_num = 0.0
            total_den = 0.0
            for deployment_index in selected:
                local = locals_[deployment_index]
                indices = rng.integers(0, len(local), len(local))
                total_num += local[numerator].to_numpy()[indices].sum()
                total_den += local[denominator].to_numpy()[indices].sum()
            values.append(
                total_num / total_den if total_den else math.nan
            )
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if not len(finite):
        return math.nan, math.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def summarize(frame, args):
    groups = [
        "scenario", "report_behavior", "benign_delta_ms",
        "uncertainty_scale", "policy", "bound_mode",
    ]
    rows = []
    for group_index, (keys, group) in enumerate(
        frame.groupby(groups, dropna=False, sort=True)
    ):
        row = dict(zip(groups, keys))
        row["trials"] = len(group)
        row["deployments"] = group.deployment.nunique()
        for metric in [
            "raw_strict_edge", "profile_exists", "capacity_unsafe",
            "trace_attack_feasible", "trace_feasible_candidate_count",
            "deferred_edge", "retained_edge", "actual_honest_support",
            "profile_support", "capacity_max_honest",
            "actual_corruption_set_covered",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
        denominator = float(group.raw_strict_edge.sum())
        row["deferred_given_edge"] = (
            float(group.deferred_edge.sum() / denominator)
            if denominator else math.nan
        )
        low, high = bootstrap_ratio(
            group, "deferred_edge", "raw_strict_edge",
            args.seed + 7001 + group_index,
            args.bootstrap_repetitions,
        )
        row["deferred_given_edge_ci95_low"] = low
        row["deferred_given_edge_ci95_high"] = high
        with_all_trials = group.assign(all_trials=1)
        low, high = bootstrap_ratio(
            with_all_trials, "deferred_edge", "all_trials",
            args.seed + 9001 + group_index,
            args.bootstrap_repetitions,
        )
        row["deferred_edge_mean_ci95_low"] = low
        row["deferred_edge_mean_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary, output_dir, args):
    primary = summary[summary.uncertainty_scale == args.primary_scale]
    window = WindowTrace(args.ping_file)
    if window.sample_values is None:
        sampling_description = (
            "Each transmission is sampled from the selected hour's truncated "
            "normal distribution `N(avg RTT/2, (mdev RTT/2)^2)`, bounded by "
            "`[min RTT/2, max RTT/2]`."
        )
    else:
        sampling_description = (
            "Each transmission is sampled from the selected AWS RTT/2 sample "
            "partition. Even-index samples form frozen history and odd-index "
            "samples form execution."
        )
    lines = [
        "# Corruption-Aware Guard Result",
        "",
        "The baseline is `n=21, f=5`, global history-optimized corrupted "
        "replicas, direct injection, plain dissemination, `rho=0`, and "
        "pair-swap Byzantine reports. A robust normal explanation requires "
        "one ingress and one release offset to fit 11 of 16 reports.",
        f"Guard bounds use mode `{args.bound_mode}`.",
        sampling_description + " The primary guard uses the complete measured "
        "interval (uncertainty scale 1).",
        "",
        "## Primary measured-window result",
        "",
        "| Scenario | Reports | Policy | Raw edge | Deferred given edge (95% CI) | Deferred over all trials (95% CI) | Retained edge |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in primary.iterrows():
        delta = "" if pd.isna(row.benign_delta_ms) else f" delta={row.benign_delta_ms:g}ms"
        scenario = f"{row.scenario}{delta}"
        conditional = (
            "n/a" if pd.isna(row.deferred_given_edge)
            else (
                f"{100 * row.deferred_given_edge:.1f}% "
                f"[{100 * row.deferred_given_edge_ci95_low:.1f}, "
                f"{100 * row.deferred_given_edge_ci95_high:.1f}]"
            )
        )
        unconditional = (
            f"{100 * row.deferred_edge_mean:.1f}% "
            f"[{100 * row.deferred_edge_mean_ci95_low:.1f}, "
            f"{100 * row.deferred_edge_mean_ci95_high:.1f}]"
        )
        lines.append(
            f"| {scenario} | {row.report_behavior} | {row.policy} | "
            f"{100 * row.raw_strict_edge_mean:.1f}% | {conditional} | "
            f"{unconditional} | "
            f"{100 * row.retained_edge_mean:.1f}% |"
        )
    attack_bound = primary[
        primary.scenario.eq("attack-fastest")
        & primary.policy.eq("profile+trace-bound")
    ].iloc[0]
    lines.extend([
        "",
        f"The actual five-replica corruption set satisfies the frozen physical "
        f"bound in {100 * attack_bound.actual_corruption_set_covered_mean:.1f}% "
        "of attack trials.",
        "",
        "`profile-only` uses only the spatial normal-profile test. "
        "`profile+trace-bound` is the corrected transaction-specific rule: "
        "the same candidate corruption set must causally reproduce the current "
        "honest report gaps under non-negative per-connection pacing. "
        "`profile+global-capacity` is a conservative "
        "edge-admission rule: it defers an otherwise normal-looking edge when "
        "some five-replica corruption set can still create a strict edge under "
        "the measured path envelope. Deferral means moving F to the next epoch, "
        "not merely deleting the direct graph edge.",
        "",
        "The three policies separate spatial anomaly detection, current-trace "
        "causal feasibility, and a network-wide safety ceiling. The corrected "
        "trace rule should be used for the detection/false-positive operating "
        "point. The global-capacity predicate remains a safety certificate, not "
        "transaction-specific evidence of malicious behavior.",
    ])
    with open(os.path.join(output_dir, "RESULTS.md"), "w") as file:
        file.write("\n".join(lines) + "\n")


def run(args):
    if args.workers < 1 or args.workers > 5:
        raise ValueError("workers must be between 1 and 5")
    os.makedirs(args.output_dir, exist_ok=True)
    trial_path = os.path.join(args.output_dir, "trial_results.csv")
    if args.summarize_only:
        if not os.path.exists(trial_path):
            raise FileNotFoundError(trial_path)
        frame = pd.read_csv(trial_path)
    else:
        tasks = [
            (deployment, vars(args)) for deployment in range(args.deployments)
        ]
        rows = []
        if args.workers == 1:
            iterator = map(run_deployment, tasks)
            for result in tqdm(
                iterator, total=len(tasks), desc="guard deployments"
            ):
                rows.extend(result)
        else:
            with Pool(args.workers) as pool:
                for result in tqdm(
                    pool.imap(run_deployment, tasks), total=len(tasks),
                    desc="guard deployments",
                ):
                    rows.extend(result)
        frame = pd.DataFrame(rows)
        frame.to_csv(trial_path, index=False)
    summary = summarize(frame, args)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    window = WindowTrace(args.ping_file)
    if window.sample_values is None:
        latency_model = (
            "hour-coherent WP snapshot; each transmission samples a normal "
            "distribution with avg RTT/2 and mdev RTT/2, truncated to the "
            "selected-hour min/max RTT/2 envelope; guard bounds use "
            f"{args.bound_mode} min/max RTT/2"
        )
    else:
        latency_model = (
            "AWS even-index RTT/2 samples are frozen history and odd-index "
            "RTT/2 samples are execution; same-hour uses execution-partition "
            "bounds and previous-hour uses the frozen history partition"
        )
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump({
            "arguments": {
                key: value for key, value in vars(args).items()
                if key != "summarize_only"
            },
            "summarized_from_existing_trials": args.summarize_only,
            "claim": "local corruption-aware edge-admission simulation",
            "normal_family": "one ingress and one common release offset",
            "capacity_model": (
                "all 20,349 five-replica subsets; same corrupt replica must "
                "observe T and deliver F"
            ),
            "trace_bound": (
                "one candidate corruption set must reproduce every current "
                "honest report gap after its causal floor; candidate-corrupted "
                "reports are unconstrained; per-connection non-negative pacing "
                "is an attacker-favorable upper bound"
            ),
            "latency_model": latency_model,
            "false_positive_definition": (
                "benign single-ingress strict edge deferred to the next epoch"
            ),
        }, file, indent=2)
    write_report(summary, args.output_dir, args)


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
            "corruption-aware-guard-global-direct-snapshot00"
        ),
    )
    parser.add_argument("--deployments", type=int, default=20)
    parser.add_argument("--trials_per_deployment", type=int, default=100)
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
    parser.add_argument(
        "--benign_deltas_ms", type=lambda value: parse_list(value, float),
        default=parse_list("-20,-10,0", float),
    )
    parser.add_argument(
        "--uncertainty_scales", type=lambda value: parse_list(value, float),
        default=parse_list("0,0.5,1", float),
    )
    parser.add_argument("--primary_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--summarize_only", action="store_true")
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
