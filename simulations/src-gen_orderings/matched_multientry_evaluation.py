#!/usr/bin/env python3
"""Paired evaluation of client-entry and corrupted-replica attack models.

The experiment fixes a deployment and all non-attack randomness, then replays
single-source and multi-source counterfactuals. Depending on the selected
observation model, sources either observe T through a public client interface
or learn T only after a corrupted replica receives it. Byzantine reports are
an explicit sensitivity parameter rather than an implicit default.
"""

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


T_GEN_TIME = 500.0
EPOCH_MS = 1000.0
PROFILE_ALIAS_MARKER = "@@replica="


def parse_list(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def stable_seed(seed, *parts):
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def stable_rng(seed, *parts):
    return random.Random(stable_seed(seed, *parts))


def physical_location(location):
    """Return the measured location underlying a logical replica identity."""
    return str(location).split(PROFILE_ALIAS_MARKER, 1)[0]


def logical_replica_location(location, replica_index):
    return f"{physical_location(location)}{PROFILE_ALIAS_MARKER}{replica_index}"


def ensure_unique_replica_identities(replicas):
    """Keep co-located replicas distinct while reusing one physical profile."""
    result = []
    used = set()
    for replica_index, location in enumerate(replicas):
        identity = str(location)
        if identity in used:
            identity = logical_replica_location(location, replica_index)
        while identity in used:
            identity += "x"
        used.add(identity)
        result.append(identity)
    return result


def resolve_path(path):
    if os.path.exists(path):
        return os.path.abspath(path)
    simulation_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(simulation_dir, "data-ping", os.path.basename(path))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(path)


class LatencyTrace:
    def __init__(self, path, one_way_mode, asymmetry_bound):
        self.path = resolve_path(path)
        self.snapshot_hours = None
        self.snapshot_matrices = None
        self.snapshot_indices = None
        self.active_snapshot = {"history": None, "execution": None}
        if self.path.endswith(".npz"):
            self._init_hourly_snapshot()
            self.one_way_mode = one_way_mode
            self.asymmetry_bound = asymmetry_bound
            return

        frame = pd.read_csv(self.path)
        value_column = "avg" if "avg" in frame.columns else "time"
        frame = frame[["source", "destination", value_column]].dropna()
        frame = frame[frame[value_column] >= 0]
        self.dataset = "wp" if value_column == "avg" else "aws"
        self.locations = sorted(frame.source.astype(str).unique().tolist())
        self.samples = {"history": {}, "execution": {}}
        for pair, group in frame.groupby(["source", "destination"], sort=False):
            values = group[value_column].to_numpy(dtype=float)
            history = values[::2]
            execution = values[1::2]
            if len(execution) == 0:
                execution = history
            self.samples["history"][(str(pair[0]), str(pair[1]))] = history
            self.samples["execution"][(str(pair[0]), str(pair[1]))] = execution
        self.one_way_mode = one_way_mode
        self.asymmetry_bound = asymmetry_bound

    def _init_hourly_snapshot(self):
        with np.load(self.path, allow_pickle=False) as archive:
            self.locations = archive["locations"].astype(str).tolist()
            self.snapshot_hours = archive["hours"].astype(str).tolist()
            self.snapshot_matrices = archive["avg_owd"].astype(np.float64)
            metadata = json.loads(str(archive["metadata"]))
        self.dataset = metadata.get("dataset", "wp")
        self.location_index = {
            location: index for index, location in enumerate(self.locations)
        }
        self.snapshot_indices = {
            "history": np.arange(0, len(self.snapshot_hours), 2, dtype=int),
            "execution": np.arange(1, len(self.snapshot_hours), 2, dtype=int),
        }
        if not len(self.snapshot_indices["execution"]):
            self.snapshot_indices["execution"] = self.snapshot_indices["history"]
        self.snapshot_metadata = metadata

    @property
    def is_hourly_snapshot(self):
        return self.snapshot_matrices is not None

    def set_snapshot_context(self, phase, seed, *key):
        if not self.is_hourly_snapshot:
            return
        candidates = self.snapshot_indices[phase]
        rng = stable_rng(seed, "hourly-snapshot", phase, *key)
        self.active_snapshot[phase] = int(candidates[rng.randrange(len(candidates))])

    def active_snapshot_hour(self, phase):
        index = self.active_snapshot.get(phase)
        return "" if index is None else self.snapshot_hours[index]

    def one_way_factor(self, seed, key):
        if self.one_way_mode == "half":
            return 0.5
        if self.one_way_mode == "low":
            return 0.5 - self.asymmetry_bound
        if self.one_way_mode == "high":
            return 0.5 + self.asymmetry_bound
        if self.one_way_mode == "bounded":
            return stable_rng(seed, "asymmetry", *key).uniform(
                0.5 - self.asymmetry_bound,
                0.5 + self.asymmetry_bound,
            )
        if self.one_way_mode == "bounded_pair":
            phase, source, destination = key[:3]
            return stable_rng(
                seed, "pair-asymmetry", phase, source, destination
            ).uniform(
                0.5 - self.asymmetry_bound,
                0.5 + self.asymmetry_bound,
            )
        raise ValueError(f"unknown one-way mode: {self.one_way_mode}")

    def sample(self, phase, source, destination, seed, *key):
        source = physical_location(source)
        destination = physical_location(destination)
        if self.is_hourly_snapshot:
            snapshot = self.active_snapshot.get(phase)
            if snapshot is None:
                raise RuntimeError(
                    f"{phase} snapshot context must be selected before sampling"
                )
            source_index = self.location_index[str(source)]
            destination_index = self.location_index[str(destination)]
            delay = float(
                self.snapshot_matrices[snapshot, source_index, destination_index]
            )
            # Snapshot matrices already contain avg RTT / 2.
            factor = self.one_way_factor(
                seed, (phase, source, destination, *key)
            )
            return delay * (factor / 0.5)
        pair = (str(source), str(destination))
        values = self.samples[phase][pair]
        rng = stable_rng(seed, "latency-index", phase, source, destination, *key)
        rtt = float(values[rng.randrange(len(values))])
        return rtt * self.one_way_factor(seed, (phase, source, destination, *key))

    def historical_distance(self, source, destination):
        """Return a stable symmetric delay used only to form deployment regions."""
        source = physical_location(source)
        destination = physical_location(destination)
        if self.is_hourly_snapshot:
            source_index = self.location_index[str(source)]
            destination_index = self.location_index[str(destination)]
            forward = self.snapshot_matrices[
                self.snapshot_indices["history"], source_index, destination_index
            ]
            reverse = self.snapshot_matrices[
                self.snapshot_indices["history"], destination_index, source_index
            ]
            return 0.5 * (float(np.mean(forward)) + float(np.mean(reverse)))
        forward = self.samples["history"][(str(source), str(destination))]
        reverse = self.samples["history"][(str(destination), str(source))]
        return 0.25 * (float(np.mean(forward)) + float(np.mean(reverse)))


def make_entries(location_order, max_apis):
    entries = []
    for index in range(max_apis):
        entries.append({
            "id": f"E{index:03d}",
            "location": location_order[index % len(location_order)],
        })
    return entries


def make_corrupted_direct_sources(replicas, corrupted):
    return [
        {
            "id": f"C{source_index:03d}",
            "location": replicas[replica_index],
            "source_replica_index": replica_index,
            "source_kind": "corrupted-replica-direct",
        }
        for source_index, replica_index in enumerate(corrupted)
    ]


def make_clustered_replicas(trace, rng, num_nodes, region_count):
    """Place distinct replicas near latency medoids and identify one per region."""
    medoids = [rng.choice(trace.locations)]
    while len(medoids) < region_count:
        medoids.append(max(
            (location for location in trace.locations if location not in medoids),
            key=lambda location: (
                min(
                    trace.historical_distance(location, medoid)
                    for medoid in medoids
                ),
                str(location),
            ),
        ))

    assigned = {region: [] for region in range(region_count)}
    for location in trace.locations:
        region = min(
            range(region_count),
            key=lambda candidate: (
                trace.historical_distance(location, medoids[candidate]),
                candidate,
            ),
        )
        assigned[region].append(location)
    for region, medoid in enumerate(medoids):
        if medoid not in assigned[region]:
            assigned[region].append(medoid)

    quotas = [num_nodes // region_count] * region_count
    for region in range(num_nodes % region_count):
        quotas[region] += 1

    replicas = []
    replica_regions = []
    corrupted_indices = []
    used = set()
    for region, quota in enumerate(quotas):
        ordered = sorted(
            assigned[region],
            key=lambda location: (
                location != medoids[region],
                trace.historical_distance(location, medoids[region]),
                str(location),
            ),
        )
        ordered.extend(sorted(
            (location for location in trace.locations if location not in assigned[region]),
            key=lambda location: (
                trace.historical_distance(location, medoids[region]),
                str(location),
            ),
        ))
        chosen = []
        for location in ordered:
            if location in used:
                continue
            chosen.append(location)
            used.add(location)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError("not enough unique locations for clustered replicas")
        corrupted_indices.append(len(replicas))
        replicas.extend(chosen)
        replica_regions.extend([region] * quota)
    return replicas, replica_regions, medoids, corrupted_indices


def optimize_corrupted_locations(trace, deployment, corrupted, args):
    """Choose fixed corrupt-replica locations using history snapshots only."""
    replicas = deployment["replicas"]
    if getattr(args, "allow_corrupted_region_colocation", False):
        candidates = list(trace.locations)
    else:
        honest_locations = {
            physical_location(location) for index, location in enumerate(replicas)
            if index not in corrupted
        }
        candidates = [
            location for location in trace.locations
            if location not in honest_locations
        ]
    if not candidates:
        raise ValueError("no candidate locations for corrupt replicas")

    quorum_size = args.num_nodes - (args.num_nodes - 1) // 4
    selected_honest = [
        index for index in deployment["reporter_order"][:quorum_size]
        if index not in corrupted
    ]
    reporter_locations = [replicas[index] for index in selected_honest]
    selected_corrupted = len(
        set(deployment["reporter_order"][:quorum_size]) & set(corrupted)
    )
    report_byzantine = min(
        getattr(args, "byzantine_reports", selected_corrupted),
        selected_corrupted,
    )
    strict_required = strict_honest_support_required(
        args.num_nodes, quorum_size, report_byzantine
    )
    if args.placement_victim_scope == "all-locations":
        training_victims = list(trace.locations)
    else:
        training_victims = [
            deployment["clients"][sample_id % len(deployment["clients"])]
            for sample_id in range(args.training_snapshots)
        ]
    arrivals = []
    targets = []
    for sample_id, victim in enumerate(training_victims):
        if trace.is_hourly_snapshot:
            trace.set_snapshot_context(
                "history", args.seed, "placement", deployment["deployment"],
                sample_id,
            )
            matrix = trace.snapshot_matrices[trace.active_snapshot["history"]]
            candidate_indices = np.asarray([
                trace.location_index[location] for location in candidates
            ], dtype=int)
            reporter_indices = np.asarray([
                trace.location_index[location]
                for location in reporter_locations
            ], dtype=int)
            victim_index = trace.location_index[victim]
            observation = matrix[victim_index, candidate_indices]
            direct = matrix[np.ix_(candidate_indices, reporter_indices)]
            delivery = direct
            if args.observation_model == "corrupted-replica-hybrid":
                relay_indices = np.asarray([
                    trace.location_index[location]
                    for location in deployment["relay_locations"]
                ], dtype=int)
                via_entry = np.min(
                    matrix[np.ix_(candidate_indices, relay_indices)][:, :, None]
                    + matrix[np.ix_(relay_indices, reporter_indices)][None, :, :],
                    axis=1,
                )
                delivery = np.minimum(direct, via_entry)
        else:
            key = (
                "placement", deployment["deployment"], sample_id,
            )
            observation = np.asarray([
                trace.sample(
                    "history", victim, candidate, args.seed,
                    *key, "observe", candidate,
                )
                for candidate in candidates
            ], dtype=np.float64)
            direct = np.asarray([
                [
                    trace.sample(
                        "history", candidate, reporter, args.seed,
                        *key, "deliver", candidate, reporter,
                    )
                    for reporter in reporter_locations
                ]
                for candidate in candidates
            ], dtype=np.float64)
            delivery = direct
            if args.observation_model == "corrupted-replica-hybrid":
                via_entry = np.asarray([
                    [
                        min(
                            trace.sample(
                                "history", candidate, relay, args.seed,
                                *key, "relay-in", candidate, relay,
                            )
                            + trace.sample(
                                "history", relay, reporter, args.seed,
                                *key, "relay-out", relay, reporter,
                            )
                            for relay in deployment["relay_locations"]
                        )
                        for reporter in reporter_locations
                    ]
                    for candidate in candidates
                ], dtype=np.float64)
                delivery = np.minimum(direct, via_entry)
        configured_placement_rho = getattr(args, "placement_rho_ms", None)
        placement_rho_ms = (
            args.rho_ms if configured_placement_rho is None
            else configured_placement_rho
        )
        arrivals.append(observation[:, None] + placement_rho_ms + delivery)
        if trace.is_hourly_snapshot:
            targets.append(matrix[victim_index, reporter_indices])
        else:
            targets.append(np.asarray([
                trace.sample(
                    "history", victim, reporter, args.seed,
                    *key, "target", reporter,
                )
                for reporter in reporter_locations
            ], dtype=np.float64))
    arrivals = np.stack(arrivals, axis=1)
    targets = np.stack(targets, axis=0)

    chosen = []
    remaining = list(range(len(candidates)))
    best_arrivals = np.full(targets.shape, np.inf, dtype=np.float64)
    selection_trace = []
    for _ in corrupted:
        # The original AWS artifact models committees larger than its 21
        # measured regions by reusing regional profiles. Once every physical
        # candidate has been selected, permit another logical identity to use
        # an already selected profile. Duplicate routes add report identities
        # but cannot create artificial path diversity.
        if not remaining:
            remaining = list(range(len(candidates)))
        best_candidate = None
        best_score = None
        best_combined = None
        for candidate in remaining:
            combined = np.minimum(best_arrivals, arrivals[candidate])
            support = np.sum(combined < targets, axis=1)
            score = (
                int(np.sum(support >= strict_required)),
                int(np.sum(support)),
                float(np.sum(np.maximum(support - strict_required + 1, 0))),
                candidates[candidate],
            )
            if best_score is None or score > best_score:
                best_candidate = candidate
                best_score = score
                best_combined = combined
        chosen.append(candidates[best_candidate])
        remaining.remove(best_candidate)
        best_arrivals = best_combined
        selection_trace.append({
            "location": candidates[best_candidate],
            "training_threshold_hits": best_score[0],
            "training_support_sum": best_score[1],
        })
    return chosen, {
        "candidate_count": len(candidates),
        "allows_honest_region_colocation": getattr(
            args, "allow_corrupted_region_colocation", False
        ),
        "victim_scope": args.placement_victim_scope,
        "training_cases": len(training_victims),
        "strict_required_honest": strict_required,
        "selection_trace": selection_trace,
    }


def make_deployment(trace, deployment, args):
    rng = stable_rng(args.seed, trace.dataset, "deployment", deployment)
    if args.replica_layout == "global-random":
        if args.num_nodes <= len(trace.locations):
            replicas = rng.sample(trace.locations, args.num_nodes)
        else:
            profiles = rng.choices(trace.locations, k=args.num_nodes)
            replicas = [
                logical_replica_location(profile, index)
                for index, profile in enumerate(profiles)
            ]
        replica_regions = list(range(args.num_nodes))
        region_medoids = [physical_location(value) for value in replicas]
        region_corrupted = []
    else:
        replicas, replica_regions, region_medoids, region_corrupted = (
            make_clustered_replicas(
                trace, rng, args.num_nodes, args.replica_regions
            )
        )
    clients = rng.sample(trace.locations, min(args.num_clients, len(trace.locations)))
    location_order = rng.sample(trace.locations, len(trace.locations))
    if args.replica_layout == "latency-clustered":
        corrupted = region_corrupted[:args.corrupted_observers]
        remaining = [
            index for index in range(args.num_nodes) if index not in corrupted
        ]
        reporter_order = corrupted + rng.sample(remaining, len(remaining))
    else:
        reporter_order = rng.sample(range(args.num_nodes), args.num_nodes)
        corrupted = reporter_order[:args.corrupted_observers]

    placement_info = {
        "candidate_count": len(corrupted),
        "strict_required_honest": None,
        "selection_trace": [],
    }
    placement_mode = getattr(args, "corrupted_placement", "region-medoid")
    if placement_mode == "static-random":
        if getattr(args, "allow_corrupted_region_colocation", False):
            candidates = list(trace.locations)
        else:
            honest_locations = {
                physical_location(location)
                for index, location in enumerate(replicas)
                if index not in corrupted
            }
            candidates = [
                location for location in trace.locations
                if location not in honest_locations
            ]
        if not candidates:
            raise ValueError("no static-random corruption locations")
        if len(candidates) >= len(corrupted):
            chosen_locations = rng.sample(candidates, len(corrupted))
        else:
            chosen_locations = rng.choices(candidates, k=len(corrupted))
        for replica_index, location in zip(corrupted, chosen_locations):
            replicas[replica_index] = (
                logical_replica_location(location, replica_index)
                if args.num_nodes > len(trace.locations) else location
            )
        placement_info = {
            "candidate_count": len(candidates),
            "strict_required_honest": None,
            "selection_trace": [
                {"location": value, "selection": "static-random"}
                for value in chosen_locations
            ],
        }
    elif placement_mode == "history-optimized-global":
        draft = {
            "deployment": deployment,
            "replicas": replicas,
            "clients": clients,
            "reporter_order": reporter_order,
            "relay_locations": location_order[:args.relay_entry_count],
        }
        chosen_locations, placement_info = optimize_corrupted_locations(
            trace, draft, corrupted, args
        )
        for replica_index, location in zip(corrupted, chosen_locations):
            replicas[replica_index] = (
                logical_replica_location(location, replica_index)
                if args.num_nodes > len(trace.locations) else location
            )

    replicas = ensure_unique_replica_identities(replicas)

    if args.observation_model == "corrupted-replica-direct":
        entries = make_corrupted_direct_sources(replicas, corrupted)
    elif args.observation_model == "corrupted-replica-hybrid":
        entries = make_entries(location_order, args.relay_entry_count)
        for entry in entries:
            entry["source_kind"] = "relay-entry"
        entries.extend(make_corrupted_direct_sources(replicas, corrupted))
    else:
        entries = make_entries(location_order, max(args.entry_counts))
    return {
        "dataset": trace.dataset,
        "deployment": deployment,
        "replicas": replicas,
        "clients": clients,
        "entries": entries,
        "reporter_order": reporter_order,
        "replica_regions": replica_regions,
        "region_medoids": region_medoids,
        "corrupted_replica_indices": corrupted,
        "corrupted_placement_info": placement_info,
    }


def corrupted_replica_indices(deployment, args):
    if args.observation_model not in {
        "corrupted-replica-relay", "corrupted-replica-direct",
        "corrupted-replica-hybrid",
    }:
        return []
    return deployment.get(
        "corrupted_replica_indices",
        deployment["reporter_order"][:args.corrupted_observers],
    )[:args.corrupted_observers]


def strict_honest_support_required(num_nodes, quorum_size, byzantine_reports):
    """Honest F<T reports needed for a tie-independent Themis edge."""
    f_count = (num_nodes - 1) // 4
    return max(
        f_count + 1 - byzantine_reports,
        quorum_size // 2 + 1 - byzantine_reports,
        0,
    )


def honest_replica_indices(deployment, args, report_byzantine=()):
    """Return replicas that are honest in the threat model, not just reports."""
    adversarial = set(report_byzantine)
    adversarial.update(corrupted_replica_indices(deployment, args))
    return [
        index for index in range(args.num_nodes)
        if index not in adversarial
    ]


def observe_entry(trace, deployment, victim, t_arrivals, entry, phase, args,
                  *key):
    """Return when an entry can learn T under the selected visibility model."""
    if args.observation_model == "direct-entry":
        return T_GEN_TIME + trace.sample(
            phase, victim, entry["location"], args.seed,
            *key, "observe", entry["id"],
        )

    if (
        args.observation_model == "corrupted-replica-direct"
        or entry.get("source_kind") == "corrupted-replica-direct"
    ):
        replica_index = entry["source_replica_index"]
        return t_arrivals[deployment["replicas"][replica_index]]

    replicas = deployment["replicas"]
    return min(
        t_arrivals[replicas[index]] + trace.sample(
            phase, replicas[index], entry["location"], args.seed,
            *key, "corrupt-relay", index, entry["id"],
        )
        for index in corrupted_replica_indices(deployment, args)
    )


def historical_entry_plan(trace, deployment, args):
    """Select a fixed C0 singleton and nested entry order from history only."""
    replicas = deployment["replicas"]
    clients = deployment["clients"]
    entries = deployment["entries"]
    winner_sets = {entry["id"]: set() for entry in entries}
    scores = defaultdict(float)
    for sample_id in range(args.training_snapshots):
        trace.set_snapshot_context(
            "history", args.seed, "train", deployment["deployment"], sample_id
        )
        victim = clients[sample_id % len(clients)]
        t_arrival = {
            node: T_GEN_TIME + trace.sample(
                "history", victim, node, args.seed,
                "train", deployment["deployment"], sample_id, "T", node,
            )
            for node in replicas
        }
        for entry in entries:
            observed = observe_entry(
                trace, deployment, victim, t_arrival, entry, "history", args,
                "train", deployment["deployment"], sample_id,
            )
            winners = set()
            for node_index, node in enumerate(replicas):
                arrival = observed + args.rho_ms + trace.sample(
                    "history", entry["location"], node, args.seed,
                    "train", deployment["deployment"], sample_id,
                    "submit", entry["id"], node,
                )
                if arrival < t_arrival[node]:
                    coordinate = (sample_id, node_index)
                    winners.add(coordinate)
                    winner_sets[entry["id"]].add(coordinate)
            scores[entry["id"]] += len(winners)

    by_id = {entry["id"]: entry for entry in entries}
    train_best = max(entries, key=lambda entry: (scores[entry["id"]], entry["id"]))
    uncovered = {
        (sample_id, node_index)
        for sample_id in range(args.training_snapshots)
        for node_index in range(len(replicas))
    }
    remaining = set(by_id)
    greedy_ids = []
    while remaining:
        chosen = max(
            remaining,
            key=lambda entry_id: (
                len(winner_sets[entry_id] & uncovered),
                scores[entry_id],
                entry_id,
            ),
        )
        greedy_ids.append(chosen)
        uncovered -= winner_sets[chosen]
        remaining.remove(chosen)
    return {
        "train_best": train_best["id"],
        "greedy_order": greedy_ids,
        "scores": dict(scores),
        "winner_sets": winner_sets,
    }


def generate_background(trace, deployment, trial, args):
    clients = deployment["clients"]
    replicas = deployment["replicas"]
    rng = np.random.default_rng(stable_seed(
        args.seed, trace.dataset, deployment["deployment"], trial, "background"
    ))
    assignments = np.arange(args.background_txs) % len(clients)
    rng.shuffle(assignments)
    generation = rng.exponential(
        EPOCH_MS / max(args.background_txs, 1), args.background_txs
    ).cumsum()
    direct = {node: [] for node in replicas}
    for tx_index, (client_index, gen_time) in enumerate(zip(assignments, generation)):
        tx = f"B{tx_index:03d}"
        source = clients[int(client_index)]
        for node in replicas:
            latency = trace.sample(
                "execution", source, node, args.seed,
                "trial", deployment["deployment"], trial, "background", tx, node,
            )
            direct[node].append((tx, float(gen_time) + latency))
    return direct


def trial_snapshot(trace, deployment, trial, args):
    trace.set_snapshot_context(
        "execution", args.seed, "trial", deployment["deployment"], trial
    )
    trace.set_snapshot_context(
        "history", args.seed, "profile", deployment["deployment"], trial
    )
    replicas = deployment["replicas"]
    clients = deployment["clients"]
    victim = stable_rng(
        args.seed, trace.dataset, deployment["deployment"], trial, "victim"
    ).choice(clients)
    direct = generate_background(trace, deployment, trial, args)
    t_arrivals = {}
    for node in replicas:
        arrival = T_GEN_TIME + trace.sample(
            "execution", victim, node, args.seed,
            "trial", deployment["deployment"], trial, "T", node,
        )
        t_arrivals[node] = arrival
        direct[node].append(("T", arrival))

    singleton_arrivals = {}
    for entry in deployment["entries"]:
        observed = observe_entry(
            trace, deployment, victim, t_arrivals, entry, "execution", args,
            "trial", deployment["deployment"], trial,
        )
        release = observed + args.rho_ms
        singleton_arrivals[entry["id"]] = {
            node: release + trace.sample(
                "execution", entry["location"], node, args.seed,
                "trial", deployment["deployment"], trial,
                "submit", entry["id"], node,
            )
            for node in replicas
        }

    profile_t_source = stable_rng(
        args.seed, trace.dataset, deployment["deployment"], trial, "profile-t"
    ).choice(clients)
    profile_f_candidates = [item for item in clients if item != profile_t_source]
    profile_f_source = stable_rng(
        args.seed, trace.dataset, deployment["deployment"], trial, "profile-f"
    ).choice(profile_f_candidates)
    profile_target = {}
    for node in replicas:
        hist_t = trace.sample(
            "history", profile_t_source, node, args.seed,
            "trial", deployment["deployment"], trial, "profile-t", node,
        )
        hist_f = trace.sample(
            "history", profile_f_source, node, args.seed,
            "trial", deployment["deployment"], trial, "profile-f", node,
        )
        profile_target[node] = t_arrivals[node] + args.profile_epsilon_ms + hist_f - hist_t
    snapshot = {
        "victim": victim,
        "direct_without_f": direct,
        "t_arrivals": t_arrivals,
        "singleton_arrivals": singleton_arrivals,
        "profile_target": profile_target,
        "profile_t_source": profile_t_source,
        "profile_f_source": profile_f_source,
        "execution_snapshot_hour": trace.active_snapshot_hour("execution"),
        "profile_snapshot_hour": trace.active_snapshot_hour("history"),
    }
    if "delayed" in args.disseminations:
        snapshot["gossip_cache"] = build_gossip_cache(
            trace, deployment, trial, args
        )
    return snapshot


def build_gossip_cache(trace, deployment, trial, args):
    replicas = deployment["replicas"]
    tx_ids = [f"B{index:03d}" for index in range(args.background_txs)] + ["T", "F"]
    tx_index = {tx: index for index, tx in enumerate(tx_ids)}
    rng = np.random.default_rng(stable_seed(
        args.seed, trace.dataset, deployment["deployment"], trial,
        "gossip-decisions-pcg64-v1",
    ))
    decisions = rng.random(
        (len(tx_ids), len(replicas), len(replicas))
    ) <= args.gossip_p
    diagonal = np.arange(len(replicas))
    decisions[:, diagonal, diagonal] = False
    return {
        "tx_index": tx_index,
        "decisions": decisions,
        "rng_scheme": "trial-seeded-pcg64-v1",
    }


def union_arrivals(singleton_arrivals, entry_ids):
    return {
        node: min(singleton_arrivals[entry_id][node] for entry_id in entry_ids)
        for node in next(iter(singleton_arrivals.values()))
    }


def apply_dissemination(trace, deployment, trial, direct_without_f, f_arrivals,
                        dissemination, args, gossip_cache=None):
    replicas = deployment["replicas"]
    direct = {
        node: list(direct_without_f[node]) + [("F", f_arrivals[node])]
        for node in replicas
    }
    if dissemination == "plain":
        return {
            node: sorted(events, key=lambda item: item[1])
            for node, events in direct.items()
        }
    if dissemination != "delayed":
        raise ValueError(dissemination)

    events = []
    sequence = 0
    for node_index, node in enumerate(replicas):
        for tx, arrival in direct[node]:
            heapq.heappush(events, (arrival, 0, node_index, sequence, tx))
            sequence += 1
    first = {node: {} for node in replicas}
    queues = [[[] for _ in replicas] for _ in replicas]
    flush_count = np.zeros((len(replicas), len(replicas)), dtype=np.int32)
    if gossip_cache is None:
        raise ValueError("delayed dissemination requires a paired gossip cache")
    tx_index = gossip_cache["tx_index"]
    decisions = gossip_cache["decisions"]
    while events:
        arrival, _, node_index, _, tx = heapq.heappop(events)
        node = replicas[node_index]
        if tx in first[node]:
            continue
        first[node][tx] = arrival
        for peer_index, peer in enumerate(replicas):
            if peer_index == node_index:
                continue
            if not decisions[tx_index[tx], node_index, peer_index]:
                continue
            queue = queues[node_index][peer_index]
            queue.append(tx)
            if len(queue) < args.batch_size:
                continue
            flush_index = int(flush_count[node_index, peer_index])
            flush_count[node_index, peer_index] += 1
            # Sample only flushes that occur. The stable key keeps every arm's
            # sender/peer/flush latency paired without precomputing unused rounds.
            latency = trace.sample(
                "execution", node, peer, args.seed,
                "trial", deployment["deployment"], trial,
                "gossip", node, peer, flush_index,
            )
            for offset, queued_tx in enumerate(queue):
                heapq.heappush(events, (
                    arrival + latency + offset * 0.001,
                    1,
                    peer_index,
                    sequence,
                    queued_tx,
                ))
                sequence += 1
            queues[node_index][peer_index] = []
    return {
        node: sorted(values.items(), key=lambda item: item[1])
        for node, values in first.items()
    }


def precedes(report, left, right):
    positions = {tx: index for index, (tx, _) in enumerate(report)}
    if left not in positions:
        return False
    return right not in positions or positions[left] < positions[right]


def forge_byzantine_report(report, mode):
    """Force F before T while controlling collateral ordering changes."""
    forged = list(report)
    if mode == "prefix":
        remainder = [item for item in forged if item[0] not in {"F", "T"}]
        return [("F", -2.0), ("T", -1.0)] + remainder
    if mode != "pair-swap":
        raise ValueError(mode)

    positions = {tx: index for index, (tx, _) in enumerate(forged)}
    if "F" not in positions or "T" not in positions:
        return forged
    f_index = positions["F"]
    t_index = positions["T"]
    if f_index < t_index:
        return forged

    # Keep every time slot and background transaction fixed; only exchange the
    # transaction labels occupying the original F and T slots.
    f_time = forged[f_index][1]
    t_time = forged[t_index][1]
    forged[t_index] = ("F", t_time)
    forged[f_index] = ("T", f_time)
    return forged


def apply_byzantine_reports(orderings, deployment, args):
    forged = [list(report) for report in orderings]
    byzantine = set(deployment["reporter_order"][:args.byzantine_reports])
    for index in byzantine:
        forged[index] = forge_byzantine_report(
            forged[index], args.byzantine_report_mode
        )
    return forged, byzantine


def pair_weights(orderings, reporters, left="F", right="T"):
    forward = sum(precedes(orderings[index], left, right) for index in reporters)
    reverse = sum(precedes(orderings[index], right, left) for index in reporters)
    presence = sum(
        left in {tx for tx, _ in orderings[index]}
        and right in {tx for tx, _ in orderings[index]}
        for index in reporters
    )
    return forward, reverse, presence


def presence_state(occurrence, num_nodes):
    f_count = (num_nodes - 1) // 4
    if occurrence >= num_nodes - 2 * f_count:
        return "solid"
    if occurrence >= f_count + 1:
        return "shaded"
    return "blank"


def build_graph(orderings, reporters, threshold, tx_order=None):
    if tx_order is None:
        first_report = orderings[reporters[0]]
        tx_list = [tx for tx, _ in first_report]
        for reporter in reporters[1:]:
            for tx, _ in orderings[reporter]:
                if tx not in tx_list:
                    tx_list.append(tx)
    else:
        tx_list = list(tx_order)
    index = {tx: position for position, tx in enumerate(tx_list)}
    count = len(tx_list)
    ranks = np.full((len(reporters), count), count + 1, dtype=np.int32)
    for row, reporter in enumerate(reporters):
        for position, (tx, _) in enumerate(orderings[reporter]):
            ranks[row, index[tx]] = position
    weights = np.zeros((count, count), dtype=np.int16)
    for row in ranks:
        present = row <= count
        before = row[:, None] < row[None, :]
        left_only = present[:, None] & ~present[None, :]
        weights += (before | left_only).astype(np.int16)
    adjacency = np.zeros((count, count), dtype=np.bool_)
    for left in range(count - 1):
        for right in range(left + 1, count):
            if weights[left, right] >= weights[right, left] and weights[left, right] >= threshold:
                adjacency[left, right] = True
            elif weights[right, left] > weights[left, right] and weights[right, left] >= threshold:
                adjacency[right, left] = True
    return tx_list, weights, adjacency


def tarjan_scc(adjacency):
    count = len(adjacency)
    number = 0
    stack = []
    on_stack = [False] * count
    indices = [-1] * count
    low = [0] * count
    components = []

    def visit(node):
        nonlocal number
        indices[node] = low[node] = number
        number += 1
        stack.append(node)
        on_stack[node] = True
        for peer in np.flatnonzero(adjacency[node]):
            peer = int(peer)
            if indices[peer] == -1:
                visit(peer)
                low[node] = min(low[node], low[peer])
            elif on_stack[peer]:
                low[node] = min(low[node], indices[peer])
        if low[node] == indices[node]:
            component = []
            while True:
                peer = stack.pop()
                on_stack[peer] = False
                component.append(peer)
                if peer == node:
                    break
            components.append(component)

    for node in range(count):
        if indices[node] == -1:
            visit(node)
    return components


def themis_hamiltonian_cycle(component, adjacency):
    """Port of Aequitas::TopologyGraph::hami/find_hamilton."""
    if len(component) <= 2:
        return list(component)
    vertices = [vertex + 1 for vertex in component]

    def has_edge(left, right):
        return bool(adjacency[left - 1, right - 1])

    for starting in vertices:
        next_vertex = [-1] * (len(adjacency) + 1)
        head = starting
        for current in vertices:
            if current == starting:
                continue
            if has_edge(current, head):
                next_vertex[current] = head
                head = current
            elif has_edge(head, current):
                previous = head
                position = next_vertex[head]
                while position != -1 and not has_edge(current, position):
                    previous = position
                    position = next_vertex[previous]
                next_vertex[previous] = current
                next_vertex[current] = position
            else:
                raise ValueError("SCC is not a tournament")

        left = head
        right = 0
        current = left
        while current > 0:
            if right:
                position = left
                previous = right
                while True:
                    if has_edge(current, position):
                        next_vertex[previous] = next_vertex[right]
                        if previous != right:
                            next_vertex[right] = left
                        left = position
                        right = current
                        break
                    if position == right:
                        break
                    previous = position
                    position = next_vertex[position]
            elif has_edge(current, left):
                right = current
            current = next_vertex[current]
        if right == 0:
            continue
        next_vertex[right] = left

        path = []
        current = next_vertex[left]
        while current != -1:
            path.append(current - 1)
            if current == left:
                break
            current = next_vertex[current]
        if (
            len(path) == len(component)
            and adjacency[path[-1], path[0]]
        ):
            return path
    raise ValueError("Themis Hamiltonian cycle construction failed")


def public_permutation(transaction_count, public_seed):
    permutation = list(range(transaction_count))
    stable_rng(public_seed, "themis-public-permutation").shuffle(permutation)
    return permutation


def finalize_graph(tx_list, adjacency, public_seed=None, permutation=None):
    if permutation is None:
        if public_seed is None:
            raise ValueError("public_seed or permutation is required")
        permutation = public_permutation(len(tx_list), public_seed)
    else:
        permutation = list(permutation)
        if sorted(permutation) != list(range(len(tx_list))):
            raise ValueError("invalid public permutation")
    permuted = adjacency[np.ix_(permutation, permutation)]
    components = tarjan_scc(permuted)
    component_of = {}
    for component_id, component in enumerate(components):
        for vertex in component:
            component_of[vertex] = component_id
    dag = [set() for _ in components]
    indegree = [0] * len(components)
    for left in range(len(permuted)):
        for right in np.flatnonzero(permuted[left]):
            source = component_of[left]
            destination = component_of[int(right)]
            if source != destination and destination not in dag[source]:
                dag[source].add(destination)
                indegree[destination] += 1
    queue = deque(index for index, value in enumerate(indegree) if value == 0)
    output = []
    while queue:
        component_id = queue.popleft()
        path = themis_hamiltonian_cycle(components[component_id], permuted)
        output.extend(permutation[vertex] for vertex in path)
        for destination in sorted(dag[component_id]):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                queue.append(destination)
    names = [tx_list[index] for index in output]
    f_index = next(index for index, tx in enumerate(tx_list) if tx == "F")
    t_index = next(index for index, tx in enumerate(tx_list) if tx == "T")
    same_scc = component_of[permutation.index(f_index)] == component_of[permutation.index(t_index)]
    scc_size = len(components[component_of[permutation.index(f_index)]]) if same_scc else 0
    return {
        "final_ft": int(names.index("F") < names.index("T")),
        "same_scc": int(same_scc),
        "ft_scc_size": scc_size,
    }


def reverse_path(adjacency, source, destination):
    queue = [source]
    seen = {source}
    while queue:
        node = queue.pop()
        if node == destination:
            return True
        for peer in np.flatnonzero(adjacency[node]):
            peer = int(peer)
            if peer not in seen:
                seen.add(peer)
                queue.append(peer)
    return False


def select_reporters(deployment, orderings, policy, quorum_size, byzantine):
    if policy == "implementation":
        return deployment["reporter_order"][:quorum_size]
    front = []
    back = []
    for index, report in enumerate(orderings):
        if index in byzantine:
            continue
        if precedes(report, "F", "T"):
            front.append(index)
        else:
            back.append(index)
    if policy == "favorable":
        return (sorted(byzantine) + front + back)[:quorum_size]
    if policy == "worst":
        return (back + front + sorted(byzantine))[:quorum_size]
    raise ValueError(policy)


def evaluate_arm(trace, deployment, trial, snapshot, arm_name, entry_ids,
                 dissemination, reporter_policy, args, run_final=True):
    f_arrivals = union_arrivals(snapshot["singleton_arrivals"], entry_ids)
    orderings_by_name = apply_dissemination(
        trace, deployment, trial, snapshot["direct_without_f"], f_arrivals,
        dissemination, args, snapshot.get("gossip_cache"),
    )
    orderings = [orderings_by_name[node] for node in deployment["replicas"]]
    orderings, byzantine = apply_byzantine_reports(
        orderings, deployment, args
    )
    quorum_size = args.num_nodes - (args.num_nodes - 1) // 4
    threshold = (args.num_nodes - 1) // 4 + 1
    reporters = select_reporters(
        deployment, orderings, reporter_policy, quorum_size, byzantine
    )
    w_ft, w_tf, pair_presence = pair_weights(orderings, reporters)
    f_occurrence = sum(
        "F" in {tx for tx, _ in orderings[index]} for index in reporters
    )
    t_occurrence = sum(
        "T" in {tx for tx, _ in orderings[index]} for index in reporters
    )
    all_transactions = {
        tx for reporter in reporters for tx, _ in orderings[reporter]
    }
    pair_order = sorted(
        all_transactions,
        key=lambda tx: stable_seed(
            args.seed, trace.dataset, deployment["deployment"], trial,
            "pair-processing-order", tx,
        ),
    )
    pair_order_ft = pair_order.index("F") < pair_order.index("T")
    direct_ft = int(
        w_ft >= threshold
        and (w_ft > w_tf or (w_ft == w_tf and pair_order_ft))
    )
    direct_tf = int(
        w_tf >= threshold
        and (w_tf > w_ft or (w_ft == w_tf and not pair_order_ft))
    )
    strict_direct_ft = int(w_ft >= threshold and w_ft > w_tf)
    has_reverse_path = math.nan
    final_values = []
    scc_values = []
    scc_sizes = []
    if run_final:
        tx_list, _, adjacency = build_graph(
            orderings, reporters, threshold, pair_order
        )
        f_index = tx_list.index("F")
        t_index = tx_list.index("T")
        # Recompute from the complete graph as a conformance assertion.
        assert direct_ft == int(adjacency[f_index, t_index])
        assert direct_tf == int(adjacency[t_index, f_index])
        has_reverse_path = int(reverse_path(adjacency, t_index, f_index))
        for repetition in range(args.public_repetitions):
            outcome = finalize_graph(
                tx_list, adjacency,
                stable_seed(args.seed, trace.dataset, deployment["deployment"],
                            trial, "public", repetition),
            )
            final_values.append(outcome["final_ft"])
            scc_values.append(outcome["same_scc"])
            scc_sizes.append(outcome["ft_scc_size"])
    honest_indices = honest_replica_indices(deployment, args, byzantine)
    direct_capacity = sum(
        f_arrivals[node] < snapshot["t_arrivals"][node]
        for index, node in enumerate(deployment["replicas"])
        if index in honest_indices
    )
    total_capacity = sum(
        precedes(orderings[index], "F", "T") for index in honest_indices
    )
    selected_direct_capacity = sum(
        f_arrivals[deployment["replicas"][index]]
        < snapshot["t_arrivals"][deployment["replicas"][index]]
        for index in reporters
        if index in honest_indices
    )
    selected_capacity = sum(
        precedes(orderings[index], "F", "T")
        for index in reporters if index in honest_indices
    )
    robust_front = sum(
        f_arrivals[node] + args.clock_epsilon_ms < snapshot["t_arrivals"][node]
        for index, node in enumerate(deployment["replicas"])
        if index in honest_indices
    )
    possible_front = sum(
        f_arrivals[node] - args.clock_epsilon_ms < snapshot["t_arrivals"][node]
        for index, node in enumerate(deployment["replicas"])
        if index in honest_indices
    )
    profile_target = snapshot["profile_target"]
    c1_order = sum(
        profile_target[node] >= snapshot["t_arrivals"][node]
        or f_arrivals[node] < snapshot["t_arrivals"][node]
        for index, node in enumerate(deployment["replicas"])
        if index in honest_indices
    )
    c1_ambush = sum(
        f_arrivals[node] <= profile_target[node] < snapshot["t_arrivals"][node]
        for index, node in enumerate(deployment["replicas"])
        if index in honest_indices
    )
    entry_map = {entry["id"]: entry["location"] for entry in deployment["entries"]}
    locations = {
        physical_location(entry_map[entry_id]) for entry_id in entry_ids
    }
    observer_indices = corrupted_replica_indices(deployment, args)
    observer_locations = [
        physical_location(deployment["replicas"][index])
        for index in observer_indices
    ]
    source_kind = {
        "corrupted-replica-direct": "corrupted-replica",
        "corrupted-replica-hybrid": "corrupted-replica-plus-relay",
    }.get(args.observation_model, "submission-api")
    selected_entry_ids = set(entry_ids)
    api_count = sum(
        entry.get("source_kind") != "corrupted-replica-direct"
        for entry in deployment["entries"]
        if entry["id"] in selected_entry_ids
    )
    return {
        "dataset": trace.dataset,
        "deployment": deployment["deployment"],
        "trial": trial,
        "dissemination": dissemination,
        "reporter_policy": reporter_policy,
        "replica_layout": args.replica_layout,
        "replica_region_count": len(set(deployment["replica_regions"])),
        "observation_model": args.observation_model,
        "corrupted_observers": args.corrupted_observers,
        "corrupted_observer_ids": ";".join(str(index) for index in observer_indices),
        "corrupted_observer_locations": ";".join(observer_locations),
        "corrupted_placement": args.corrupted_placement,
        "b": args.byzantine_reports,
        "byzantine_report_mode": args.byzantine_report_mode,
        "rho_ms": args.rho_ms,
        "one_way_mode": args.one_way_mode,
        "asymmetry_bound": args.asymmetry_bound,
        "clock_epsilon_ms": args.clock_epsilon_ms,
        "arm": arm_name,
        "entry_count": len(entry_ids),
        "attack_source_kind": source_kind,
        "attack_source_count": len(entry_ids),
        "api_count": api_count,
        "unique_region_count": len(locations),
        "entry_ids": ";".join(entry_ids),
        "entry_locations": ";".join(sorted(locations)),
        "victim": snapshot["victim"],
        "execution_snapshot_hour": snapshot["execution_snapshot_hour"],
        "profile_snapshot_hour": snapshot["profile_snapshot_hour"],
        "reporter_ids": ";".join(str(index) for index in reporters),
        "pair_presence": pair_presence,
        "f_occurrence": f_occurrence,
        "t_occurrence": t_occurrence,
        "f_presence_state": presence_state(f_occurrence, args.num_nodes),
        "t_presence_state": presence_state(t_occurrence, args.num_nodes),
        "direct_honest_capacity": direct_capacity,
        "post_dissemination_honest_capacity": total_capacity,
        "total_honest_capacity": total_capacity,
        "selected_direct_honest_capacity": selected_direct_capacity,
        "selected_honest_capacity": selected_capacity,
        "robust_front_capacity": robust_front,
        "possible_front_capacity": possible_front,
        "c1_profile_order_capacity": c1_order,
        "c1_profile_ambush_capacity": c1_ambush,
        "w_FT": w_ft,
        "w_TF": w_tf,
        "pair_tie": int(w_ft == w_tf),
        "pair_processing_first": "F" if pair_order_ft else "T",
        "fairupdate_applied": 0,
        "direct_edge_FT": direct_ft,
        "strict_direct_edge_FT": strict_direct_ft,
        "direct_edge_TF": direct_tf,
        "edge_and_final_success": (
            direct_ft * float(np.mean(final_values)) if final_values else math.nan
        ),
        "strict_edge_and_final_success": (
            strict_direct_ft * float(np.mean(final_values))
            if final_values else math.nan
        ),
        "reverse_path_T_to_F": has_reverse_path,
        "same_scc_rate": float(np.mean(scc_values)) if scc_values else math.nan,
        "mean_ft_scc_size": float(np.mean(scc_sizes)) if scc_sizes else math.nan,
        "final_F_before_T_rate": float(np.mean(final_values)) if final_values else math.nan,
        "deterministic_edge_to_output": (
            int(direct_ft and not has_reverse_path)
            if not math.isnan(has_reverse_path) else math.nan
        ),
        "public_repetitions": len(final_values),
    }


def jaccard(left, right):
    union = left | right
    return len(left & right) / len(union) if union else math.nan


def hierarchical_bootstrap(group, metric, seed, repetitions):
    collapsed = group.groupby(["deployment", "trial"], as_index=False)[metric].mean()
    deployments = sorted(collapsed.deployment.unique())
    rng = np.random.default_rng(seed)
    deployment_bootstrap = []
    for deployment in deployments:
        values = collapsed[collapsed.deployment == deployment][metric].to_numpy(dtype=float)
        indices = rng.integers(0, len(values), size=(repetitions, len(values)))
        deployment_bootstrap.append(values[indices].mean(axis=1))
    deployment_bootstrap = np.asarray(deployment_bootstrap)
    sampled_deployments = rng.integers(
        0, len(deployments), size=(repetitions, len(deployments))
    )
    repetition = np.arange(repetitions)[:, None]
    estimates = deployment_bootstrap[sampled_deployments, repetition].mean(axis=1)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def summarize(rows, args):
    frame = pd.DataFrame(rows)
    metrics = [
        "unique_region_count",
        "direct_honest_capacity", "post_dissemination_honest_capacity",
        "total_honest_capacity", "selected_direct_honest_capacity",
        "selected_honest_capacity",
        "robust_front_capacity", "possible_front_capacity",
        "direct_edge_FT", "strict_direct_edge_FT", "same_scc_rate",
        "final_F_before_T_rate", "edge_and_final_success",
        "strict_edge_and_final_success",
        "deterministic_edge_to_output", "c1_profile_order_capacity",
        "c1_profile_ambush_capacity",
    ]
    groups = [
        "dataset", "dissemination", "reporter_policy",
        "replica_layout", "replica_region_count", "observation_model",
        "corrupted_observers", "corrupted_placement", "b",
        "byzantine_report_mode", "arm",
        "entry_count", "rho_ms",
        "one_way_mode", "clock_epsilon_ms",
    ]
    output = []
    for group_index, (keys, group) in enumerate(frame.groupby(groups, dropna=False, sort=True)):
        row = dict(zip(groups, keys))
        row["rows"] = len(group)
        row["deployments"] = group.deployment.nunique()
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            low, high = hierarchical_bootstrap(
                group.dropna(subset=[metric]), metric,
                args.seed + 700001 + group_index * 101,
                args.bootstrap_repetitions,
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        output.append(row)
    return pd.DataFrame(output)


def summarize_separation(frame, args):
    if frame.empty:
        return pd.DataFrame()
    groups = [
        "dataset", "dissemination", "reporter_policy",
        "replica_layout", "replica_region_count", "observation_model",
        "corrupted_observers", "corrupted_placement", "b",
        "byzantine_report_mode", "rho_ms",
    ]
    metrics = [
        "all_singletons_direct_fail_union_succeeds",
        "all_singletons_final_fail_union_succeeds",
        "max_singleton_capacity", "union_capacity",
        "capacity_gain", "direct_edge_gain", "final_rate_gain",
    ]
    rows = []
    for group_index, (keys, group) in enumerate(frame.groupby(groups, sort=True)):
        row = dict(zip(groups, keys))
        row["trials"] = len(group)
        row["deployments"] = group.deployment.nunique()
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            low, high = hierarchical_bootstrap(
                group.dropna(subset=[metric]), metric,
                args.seed + 810001 + group_index * 131,
                args.bootstrap_repetitions,
            )
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary, output_dir):
    if plt is None or summary.empty:
        return
    selected = summary[
        summary.arm.isin(["random_subset", "history_greedy_subset", "full_union"])
        & (summary.reporter_policy == "implementation")
    ].copy()
    if selected.empty or not (selected.arm == "random_subset").any():
        return
    datasets = sorted(selected.dataset.unique())
    fig, axes = plt.subplots(
        len(datasets), 2, figsize=(10.4, 3.4 * len(datasets)),
        constrained_layout=True, sharex=True, sharey=True, squeeze=False,
    )
    colors = {"plain": "#176b5b", "delayed": "#c26418"}
    for row_index, dataset in enumerate(datasets):
        dataset_panel = selected[selected.dataset == dataset]
        entry_counts = sorted(
            dataset_panel.loc[
                dataset_panel.arm == "random_subset", "entry_count"
            ].unique()
        )
        positions = np.arange(len(entry_counts))
        for dissemination, panel in dataset_panel.groupby("dissemination"):
            random_panel = (
                panel[panel.arm == "random_subset"]
                .set_index("entry_count").loc[entry_counts].reset_index()
            )
            for column, metric in enumerate(
                ["direct_edge_FT", "final_F_before_T_rate"]
            ):
                mean = random_panel[f"{metric}_mean"]
                axes[row_index, column].errorbar(
                    positions, mean,
                    yerr=[
                        mean - random_panel[f"{metric}_ci95_low"],
                        random_panel[f"{metric}_ci95_high"] - mean,
                    ],
                    marker="o", linewidth=2, color=colors[dissemination],
                    capsize=2, label=dissemination,
                )
        axes[row_index, 0].set_ylabel(
            f"{dataset.upper()}\nPaired trial probability"
        )
        for column in range(2):
            axis = axes[row_index, column]
            axis.set_xticks(positions, [str(value) for value in entry_counts])
            axis.set_ylim(-0.03, 1.03)
            axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_title("Direct F -> T edge", fontweight="bold")
    axes[0, 1].set_title("F before T in final output", fontweight="bold")
    for axis in axes[-1, :]:
        axis.set_xlabel("Submission APIs (nested random set)")
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Client-only matched multi-entry attack (b = 0)", fontweight="bold")
    for extension in ("png", "svg"):
        fig.savefig(os.path.join(output_dir, f"matched_attack_success.{extension}"), dpi=180)
    plt.close(fig)


def plot_overlap(overlaps, output_dir):
    if plt is None or overlaps.empty:
        return
    datasets = sorted(overlaps.dataset.unique())
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.2 * len(datasets), 3.4),
        constrained_layout=True, squeeze=False,
    )
    for axis, dataset in zip(axes.flat, datasets):
        panel = overlaps[(overlaps.dataset == dataset) & (overlaps["union"] > 0)]
        axis.hist(panel.jaccard.dropna(), bins=np.linspace(0, 1, 21),
                  color="#2f6b59", edgecolor="white")
        axis.set_title(dataset.upper())
        axis.set_xlabel("Winner-set Jaccard (non-empty union)")
        axis.set_ylabel("Entry pairs")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Singleton geographic complementarity", fontweight="bold")
    for extension in ("png", "svg"):
        fig.savefig(os.path.join(output_dir, f"winner_set_overlap.{extension}"), dpi=180)
    plt.close(fig)


def plot_separation(separation, output_dir):
    if plt is None or separation.empty:
        return
    labels = [
        f"{row.dataset.upper()}\n{row.dissemination}"
        for _, row in separation.iterrows()
    ]
    x = np.arange(len(labels))
    has_final = (
        "all_singletons_final_fail_union_succeeds_mean" in separation.columns
    )
    width = 0.34 if has_final else 0.5
    fig, axis = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    direct = separation.all_singletons_direct_fail_union_succeeds_mean
    axis.bar(
        x - width / 2 if has_final else x,
        direct, width, color="#176b5b", label="Direct-edge separation",
        yerr=[
            direct
            - separation.all_singletons_direct_fail_union_succeeds_ci95_low,
            separation.all_singletons_direct_fail_union_succeeds_ci95_high
            - direct,
        ], capsize=3,
    )
    if has_final:
        final = separation.all_singletons_final_fail_union_succeeds_mean
        axis.bar(
            x + width / 2,
            final, width, color="#c26418", label="Final-output separation",
            yerr=[
                final
                - separation.all_singletons_final_fail_union_succeeds_ci95_low,
                separation.all_singletons_final_fail_union_succeeds_ci95_high
                - final,
            ], capsize=3,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Paired trial probability")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("All singletons fail while the full union succeeds", fontweight="bold")
    for extension in ("png", "svg"):
        fig.savefig(os.path.join(output_dir, f"singleton_union_separation.{extension}"), dpi=180)
    plt.close(fig)


def write_metadata(output_dir, args, traces, deployments):
    hourly = any(trace.is_hourly_snapshot for trace in traces)
    claim = {
        "direct-entry": "paired client-entry singleton-versus-union evaluation",
        "corrupted-replica-relay": (
            "paired corrupted-replica observation with external relay entries"
        ),
        "corrupted-replica-direct": (
            "paired corrupted-replica observation and direct injection"
        ),
        "corrupted-replica-hybrid": (
            "paired corrupted-replica direct-and-relay injection"
        ),
    }[args.observation_model]
    metadata = {
        "claim": claim,
        "b": args.byzantine_reports,
        "byzantine_report_mode": args.byzantine_report_mode,
        "arguments": vars(args),
        "latency_files": [trace.path for trace in traces],
        "latency_partition": (
            "even UTC hours history; odd UTC hours execution; every trial uses "
            "one coherent avg RTT/2 matrix"
            if hourly else "even samples history; odd samples execution"
        ),
        "hourly_snapshot_metadata": [
            trace.snapshot_metadata
            for trace in traces if trace.is_hourly_snapshot
        ],
        "c0_policy": {
            "direct-entry": (
                "each entry releases once after its own public observation plus rho"
            ),
            "corrupted-replica-relay": (
                "each entry releases after the first corrupted replica receives T, "
                "relays it to that entry, and rho elapses"
            ),
            "corrupted-replica-direct": (
                "each corrupted replica directly injects F after receiving T and rho"
            ),
            "corrupted-replica-hybrid": (
                "corrupted replicas inject F directly and simultaneously relay T "
                "to external entries; every recipient keeps the earliest F copy"
            ),
        }[args.observation_model],
        "history_policy": "entry selection uses training snapshots only; execution RTT is hidden",
        "reporter_selection": (
            "implementation policy uses the deployment's fixed reporter order; "
            "clustered corrupted-replica deployments place every corrupted "
            "replica first, so b>0 runs are attacker-favorable quorums"
        ),
        "weight_semantics": "left precedes right if left is earlier or left-only",
        "fairupdate_scope": "all T/F reports are present and solid; FairUpdate is recorded as not applied",
        "finalizer": "Python tournament/SCC reference; C++ conformance is a separate artifact",
        "confidence_interval": (
            "95% percentile hierarchical bootstrap; resample deployments, "
            "then trials within deployments"
        ),
        "deployments": deployments,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)


def write_conformance_case(path, orderings, reporters, pair_order,
                           public_permutations):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        file.write(f"{len(orderings)}\n")
        file.write(",".join(str(index) for index in reporters) + "\n")
        file.write("PAIR:" + ",".join(pair_order) + "\n")
        file.write(f"PUBLIC_COUNT:{len(public_permutations)}\n")
        for permutation in public_permutations:
            file.write("PERM:" + ",".join(str(index) for index in permutation) + "\n")
        for report in orderings:
            file.write(",".join(tx for tx, _ in report) + ",\n")


def run(args):
    f_count = (args.num_nodes - 1) // 4
    if args.byzantine_reports < 0 or args.byzantine_reports > f_count:
        raise ValueError(f"byzantine_reports must be between 0 and {f_count}")
    if args.corrupted_observers < 0 or args.corrupted_observers > f_count:
        raise ValueError(f"corrupted_observers must be between 0 and {f_count}")
    corrupted_models = {
        "corrupted-replica-relay", "corrupted-replica-direct",
        "corrupted-replica-hybrid",
    }
    if args.observation_model in corrupted_models and args.corrupted_observers == 0:
        raise ValueError(
            f"{args.observation_model} requires corrupted_observers > 0"
        )
    if (
        args.observation_model in corrupted_models
        and args.byzantine_reports > args.corrupted_observers
    ):
        raise ValueError(
            "byzantine_reports cannot exceed corrupted_observers in relay mode"
        )
    if args.replica_regions < 1 or args.replica_regions > args.num_nodes:
        raise ValueError("replica_regions must be between 1 and num_nodes")
    if args.relay_entry_count < 1:
        raise ValueError("relay_entry_count must be positive")
    if (
        args.replica_layout == "latency-clustered"
        and args.corrupted_observers > args.replica_regions
    ):
        raise ValueError(
            "one-per-region corruption requires corrupted_observers <= replica_regions"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    traces = [
        LatencyTrace(path, args.one_way_mode, args.asymmetry_bound)
        for path in args.ping_files
    ]
    result_rows = []
    singleton_rows = []
    winner_rows = []
    overlap_rows = []
    separation_rows = []
    manifests = []
    total = len(traces) * args.deployments * args.trials_per_deployment
    progress = tqdm(total=total, desc="matched trials")

    for trace in traces:
        for deployment_id in range(
            args.deployment_start, args.deployment_start + args.deployments
        ):
            deployment = make_deployment(trace, deployment_id, args)
            plan = historical_entry_plan(trace, deployment, args)
            random_orders = [
                stable_rng(
                    args.seed, trace.dataset, deployment_id,
                    "random-entry-order", repetition,
                ).sample(
                    [entry["id"] for entry in deployment["entries"]],
                    len(deployment["entries"]),
                )
                for repetition in range(args.random_subset_repetitions)
            ]
            manifests.append({
                "dataset": trace.dataset,
                "deployment": deployment_id,
                "replicas": json.dumps(deployment["replicas"]),
                "clients": json.dumps(deployment["clients"]),
                "entries": json.dumps(deployment["entries"]),
                "reporter_order": json.dumps(deployment["reporter_order"]),
                "replica_layout": args.replica_layout,
                "replica_region_count": len(set(deployment["replica_regions"])),
                "replica_region_ids": json.dumps(deployment["replica_regions"]),
                "replica_region_medoids": json.dumps(deployment["region_medoids"]),
                "observation_model": args.observation_model,
                "corrupted_observers": args.corrupted_observers,
                "corrupted_observer_ids": json.dumps(
                    corrupted_replica_indices(deployment, args)
                ),
                "corrupted_observer_locations": json.dumps([
                    deployment["replicas"][index]
                    for index in corrupted_replica_indices(deployment, args)
                ]),
                "corrupted_placement": args.corrupted_placement,
                "placement_rho_ms": (
                    args.rho_ms
                    if getattr(args, "placement_rho_ms", None) is None
                    else args.placement_rho_ms
                ),
                "byzantine_report_mode": args.byzantine_report_mode,
                "corrupted_placement_info": json.dumps(
                    deployment["corrupted_placement_info"]
                ),
                "train_best_singleton": plan["train_best"],
                "history_greedy_order": json.dumps(plan["greedy_order"]),
                "random_entry_orders": json.dumps(random_orders),
            })

            for trial in range(args.trials_per_deployment):
                snapshot = trial_snapshot(trace, deployment, trial, args)
                direct_winners = {
                    entry["id"]: {
                        index for index, node in enumerate(deployment["replicas"])
                        if snapshot["singleton_arrivals"][entry["id"]][node]
                        < snapshot["t_arrivals"][node]
                    }
                    for entry in deployment["entries"]
                }
                for entry in deployment["entries"]:
                    winners = direct_winners[entry["id"]]
                    winner_rows.append({
                        "dataset": trace.dataset,
                        "deployment": deployment_id,
                        "trial": trial,
                        "entry_id": entry["id"],
                        "entry_location": entry["location"],
                        "winner_count": len(winners),
                        "winner_indices": ";".join(str(value) for value in sorted(winners)),
                    })
                unique_entry_ids = []
                seen_locations = set()
                for entry in deployment["entries"]:
                    if entry["location"] not in seen_locations:
                        unique_entry_ids.append(entry["id"])
                        seen_locations.add(entry["location"])
                pairs = []
                for left_index in range(len(unique_entry_ids)):
                    for right_index in range(left_index + 1, len(unique_entry_ids)):
                        pairs.append((unique_entry_ids[left_index], unique_entry_ids[right_index]))
                sampled_pairs = stable_rng(
                    args.seed, trace.dataset, deployment_id, trial, "jaccard"
                ).sample(pairs, min(args.jaccard_pairs_per_trial, len(pairs)))
                overlap_rows.extend({
                    "dataset": trace.dataset,
                    "deployment": deployment_id,
                    "trial": trial,
                    "left_entry": left,
                    "right_entry": right,
                    "jaccard": jaccard(direct_winners[left], direct_winners[right]),
                    "overlap": len(direct_winners[left] & direct_winners[right]),
                    "union": len(direct_winners[left] | direct_winners[right]),
                    "both_empty": int(not direct_winners[left] and not direct_winners[right]),
                } for left, right in sampled_pairs)

                oracle_best = max(
                    deployment["entries"],
                    key=lambda entry: (len(direct_winners[entry["id"]]), entry["id"]),
                )["id"]
                selected_arms = []
                for count in args.entry_counts:
                    for repetition, random_order in enumerate(random_orders):
                        selected_arms.append((
                            "random_subset", random_order[:count], repetition
                        ))
                    selected_arms.append((
                        "history_greedy_subset", plan["greedy_order"][:count], -1
                    ))
                selected_arms.extend([
                    ("train_best_singleton", [plan["train_best"]], -1),
                    ("oracle_best_singleton", [oracle_best], -1),
                    ("full_union", [entry["id"] for entry in deployment["entries"]], -1),
                ])
                if args.selected_arm_mode == "core":
                    selected_arms = selected_arms[-3:]
                elif args.selected_arm_mode == "full_union_only":
                    selected_arms = [selected_arms[-1]]

                for dissemination in args.disseminations:
                    for reporter_policy in args.reporter_policies:
                        singleton_results = []
                        run_all = (
                            args.selected_arm_mode in {"full", "core"}
                            and (dissemination == "plain" or args.delayed_all_singletons)
                        )
                        if run_all:
                            for entry in deployment["entries"]:
                                row = evaluate_arm(
                                    trace, deployment, trial, snapshot,
                                    "singleton", [entry["id"]], dissemination,
                                    reporter_policy, args,
                                    run_final=args.final_all_singletons,
                                )
                                singleton_rows.append(row)
                                singleton_results.append(row)
                        selected_results = []
                        for arm_name, entry_ids, subset_repetition in selected_arms:
                            row = evaluate_arm(
                                trace, deployment, trial, snapshot,
                                arm_name, entry_ids, dissemination,
                                reporter_policy, args, run_final=True,
                            )
                            row["subset_repetition"] = subset_repetition
                            result_rows.append(row)
                            selected_results.append(row)
                            if (
                                args.conformance_cases > 0
                                and trial < args.conformance_cases
                                and reporter_policy == "implementation"
                                and arm_name in {"oracle_best_singleton", "full_union"}
                            ):
                                f_arrivals = union_arrivals(
                                    snapshot["singleton_arrivals"], entry_ids
                                )
                                case_by_name = apply_dissemination(
                                    trace, deployment, trial,
                                    snapshot["direct_without_f"], f_arrivals,
                                    dissemination, args,
                                    snapshot.get("gossip_cache"),
                                )
                                case_orderings = [
                                    case_by_name[node] for node in deployment["replicas"]
                                ]
                                case_orderings, _ = apply_byzantine_reports(
                                    case_orderings, deployment, args
                                )
                                case_reporters = deployment["reporter_order"][:
                                    args.num_nodes - (args.num_nodes - 1) // 4
                                ]
                                case_transactions = {
                                    tx for report in case_orderings for tx, _ in report
                                }
                                case_pair_order = sorted(
                                    case_transactions,
                                    key=lambda tx: stable_seed(
                                        args.seed, trace.dataset, deployment_id,
                                        trial, "pair-processing-order", tx,
                                    ),
                                )
                                case_name = (
                                    f"{trace.dataset}-d{deployment_id}-t{trial}-"
                                    f"{dissemination}-{arm_name}.case"
                                )
                                case_public_permutations = [
                                    public_permutation(
                                        len(case_pair_order),
                                        stable_seed(
                                            args.seed, trace.dataset,
                                            deployment_id, trial, "public",
                                            repetition,
                                        ),
                                    )
                                    for repetition in range(args.public_repetitions)
                                ]
                                write_conformance_case(
                                    os.path.join(args.output_dir, "conformance", case_name),
                                    case_orderings, case_reporters,
                                    case_pair_order, case_public_permutations,
                                )
                        union_row = next(row for row in selected_results if row["arm"] == "full_union")
                        if singleton_results:
                            max_singleton_edge = max(row["direct_edge_FT"] for row in singleton_results)
                            final_singletons = [
                                row["final_F_before_T_rate"] for row in singleton_results
                                if not math.isnan(row["final_F_before_T_rate"])
                            ]
                            max_singleton_final = max(final_singletons) if final_singletons else math.nan
                            separation_rows.append({
                                "dataset": trace.dataset,
                                "deployment": deployment_id,
                                "trial": trial,
                                "dissemination": dissemination,
                                "reporter_policy": reporter_policy,
                                "replica_layout": args.replica_layout,
                                "replica_region_count": len(set(
                                    deployment["replica_regions"]
                                )),
                                "observation_model": args.observation_model,
                                "corrupted_observers": args.corrupted_observers,
                                "corrupted_placement": args.corrupted_placement,
                                "b": args.byzantine_reports,
                                "byzantine_report_mode": args.byzantine_report_mode,
                                "rho_ms": args.rho_ms,
                                "all_singletons_direct_fail_union_succeeds": int(
                                    max_singleton_edge == 0 and union_row["direct_edge_FT"] == 1
                                ),
                                "all_singletons_final_fail_union_succeeds": (
                                    int(max_singleton_final == 0 and union_row["final_F_before_T_rate"] > 0)
                                    if not math.isnan(max_singleton_final) else math.nan
                                ),
                                "max_singleton_capacity": max(
                                    row["total_honest_capacity"] for row in singleton_results
                                ),
                                "union_capacity": union_row["total_honest_capacity"],
                                "max_singleton_direct_edge": max_singleton_edge,
                                "union_direct_edge": union_row["direct_edge_FT"],
                                "max_singleton_final_rate": max_singleton_final,
                                "union_final_rate": union_row["final_F_before_T_rate"],
                                "capacity_gain": union_row["total_honest_capacity"] - max(
                                    row["total_honest_capacity"] for row in singleton_results
                                ),
                                "direct_edge_gain": union_row["direct_edge_FT"] - max_singleton_edge,
                                "final_rate_gain": (
                                    union_row["final_F_before_T_rate"] - max_singleton_final
                                    if not math.isnan(max_singleton_final) else math.nan
                                ),
                            })
                progress.update()
    progress.close()

    results = pd.DataFrame(result_rows)
    singletons = pd.DataFrame(singleton_rows)
    winners = pd.DataFrame(winner_rows)
    overlaps = pd.DataFrame(overlap_rows)
    separations = pd.DataFrame(separation_rows)
    summary = summarize(result_rows, args)
    separation_summary = summarize_separation(separations, args)
    results.to_csv(os.path.join(args.output_dir, "trial_results.csv"), index=False)
    singletons.to_csv(
        os.path.join(args.output_dir, "singleton_results.csv.gz"),
        index=False, compression="gzip",
    )
    winners.to_csv(
        os.path.join(args.output_dir, "singleton_winner_sets.csv.gz"),
        index=False, compression="gzip",
    )
    overlaps.to_csv(os.path.join(args.output_dir, "winner_overlap.csv"), index=False)
    separations.to_csv(os.path.join(args.output_dir, "separation_trials.csv"), index=False)
    separation_summary.to_csv(
        os.path.join(args.output_dir, "separation_summary.csv"), index=False
    )
    pd.DataFrame(manifests).to_csv(
        os.path.join(args.output_dir, "deployment_manifest.csv"), index=False
    )
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    if not overlaps.empty:
        overlaps.groupby("dataset").jaccard.describe().to_csv(
            os.path.join(args.output_dir, "winner_overlap_summary.csv")
        )
        plot_overlap(overlaps, args.output_dir)
    plot_separation(separation_summary, args.output_dir)
    plot_summary(summary, args.output_dir)
    write_metadata(args.output_dir, args, traces, manifests)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ping_files", nargs="+",
        default=["./data-ping/wp-simplified.csv", "./data-ping/aws-individual_pings.csv"],
    )
    parser.add_argument("--output_dir", default="./data-causality-comparison/matched00")
    parser.add_argument("--deployments", type=int, default=3)
    parser.add_argument("--deployment_start", type=int, default=0)
    parser.add_argument("--trials_per_deployment", type=int, default=20)
    parser.add_argument("--num_nodes", type=int, default=21)
    parser.add_argument("--num_clients", type=int, default=15)
    parser.add_argument(
        "--replica_layout",
        choices=["global-random", "latency-clustered"],
        default="global-random",
    )
    parser.add_argument("--replica_regions", type=int, default=5)
    parser.add_argument("--byzantine_reports", type=int, default=0)
    parser.add_argument(
        "--byzantine_report_mode",
        choices=["pair-swap", "prefix"],
        default="pair-swap",
        help=(
            "pair-swap keeps all background slots fixed and swaps F/T only "
            "when needed; prefix reproduces the legacy [F,T]+remainder model"
        ),
    )
    parser.add_argument(
        "--observation_model",
        choices=[
            "direct-entry", "corrupted-replica-relay",
            "corrupted-replica-direct",
            "corrupted-replica-hybrid",
        ],
        default="direct-entry",
    )
    parser.add_argument("--corrupted_observers", type=int, default=0)
    parser.add_argument(
        "--corrupted_placement",
        choices=["region-medoid", "static-random", "history-optimized-global"],
        default="region-medoid",
    )
    parser.add_argument(
        "--placement_victim_scope",
        choices=["all-locations", "deployment-clients"],
        default="all-locations",
    )
    parser.add_argument(
        "--allow_corrupted_region_colocation", action="store_true",
        help=(
            "Allow a corrupted logical replica to reuse a measured region "
            "profile occupied by an honest replica. This matches the original "
            "AWS artifact's logical-node scaling convention."
        ),
    )
    parser.add_argument("--relay_entry_count", type=int, default=220)
    parser.add_argument("--background_txs", type=int, default=300)
    parser.add_argument("--entry_counts", type=lambda value: parse_list(value, int),
                        default=parse_list("1,2,4,8,16,32,64,128,220", int))
    parser.add_argument("--rho_ms", type=float, default=0.0)
    parser.add_argument(
        "--placement_rho_ms", type=float,
        help=(
            "Reaction delay assumed while training corrupted placement. "
            "Defaults to rho_ms; set this independently for a frozen-placement "
            "reaction-delay sensitivity."
        ),
    )
    parser.add_argument("--profile_epsilon_ms", type=float, default=0.0)
    parser.add_argument("--training_snapshots", type=int, default=24)
    parser.add_argument("--random_subset_repetitions", type=int, default=3)
    parser.add_argument(
        "--selected_arm_mode", choices=["full", "core", "full_union_only"],
        default="full",
        help="Use full_union_only for inexpensive reporter/sensitivity replays.",
    )
    parser.add_argument("--disseminations", type=lambda value: parse_list(value, str),
                        default=parse_list("plain,delayed", str))
    parser.add_argument("--reporter_policies", type=lambda value: parse_list(value, str),
                        default=parse_list("implementation", str))
    parser.add_argument("--gossip_p", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--delayed_all_singletons", action="store_true")
    parser.add_argument("--final_all_singletons", action="store_true")
    parser.add_argument("--public_repetitions", type=int, default=3)
    parser.add_argument("--one_way_mode", choices=[
        "half", "low", "high", "bounded", "bounded_pair",
    ],
                        default="half")
    parser.add_argument("--asymmetry_bound", type=float, default=0.15)
    parser.add_argument("--clock_epsilon_ms", type=float, default=0.0)
    parser.add_argument("--jaccard_pairs_per_trial", type=int, default=500)
    parser.add_argument("--bootstrap_repetitions", type=int, default=1000)
    parser.add_argument("--conformance_cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
