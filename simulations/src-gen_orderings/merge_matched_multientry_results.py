#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import pathlib
import shutil

import pandas as pd


MODULE_PATH = pathlib.Path(__file__).with_name("matched_multientry_evaluation.py")
SPEC = importlib.util.spec_from_file_location("matched_multientry", MODULE_PATH)
MATCHED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATCHED)


TABLES = {
    "trial_results.csv": "trial_results.csv",
    "singleton_results.csv.gz": "singleton_results.csv.gz",
    "singleton_winner_sets.csv.gz": "singleton_winner_sets.csv.gz",
    "winner_overlap.csv": "winner_overlap.csv",
    "separation_trials.csv": "separation_trials.csv",
    "deployment_manifest.csv": "deployment_manifest.csv",
}


def read_table(shards, filename):
    frames = []
    for shard in shards:
        path = os.path.join(shard, filename)
        if os.path.exists(path):
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def physical_location_list(value):
    locations = {
        MATCHED.physical_location(item)
        for item in str(value).split(";") if item
    }
    return ";".join(sorted(locations))


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    combined = {
        output: read_table(args.shards, source)
        for source, output in TABLES.items()
    }
    for output, frame in combined.items():
        if frame.empty:
            continue
        compression = "gzip" if output.endswith(".gz") else None
        frame.to_csv(
            os.path.join(args.output_dir, output), index=False,
            compression=compression,
        )

    results = combined["trial_results.csv"]
    if not results.empty:
        results["entry_locations"] = results.entry_locations.map(
            physical_location_list
        )
        results["corrupted_observer_locations"] = (
            results.corrupted_observer_locations.map(physical_location_list)
        )
        results["unique_region_count"] = results.entry_locations.map(
            lambda value: len(value.split(";")) if value else 0
        )
        results["edge_and_final_success"] = (
            results.direct_edge_FT * results.final_F_before_T_rate
        )
        results.to_csv(
            os.path.join(args.output_dir, "trial_results.csv"), index=False
        )
    summary_args = argparse.Namespace(
        seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    summary = MATCHED.summarize(results.to_dict("records"), summary_args)
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    MATCHED.plot_summary(summary, args.output_dir)

    full_union = results[
        (results.arm == "full_union")
        & (results.reporter_policy == "implementation")
    ].copy()
    if not full_union.empty:
        full_union["edge_without_reverse_path"] = (
            (full_union.direct_edge_FT == 1)
            & (full_union.reverse_path_T_to_F == 0)
        ).astype(int)
        full_union["edge_with_reverse_path"] = (
            (full_union.direct_edge_FT == 1)
            & (full_union.reverse_path_T_to_F == 1)
        ).astype(int)
        edge_rows = []
        for (dataset, dissemination), group in full_union.groupby(
            ["dataset", "dissemination"], sort=True
        ):
            no_reverse = group[group.edge_without_reverse_path == 1]
            reverse = group[group.edge_with_reverse_path == 1]
            edge_rows.append({
                "dataset": dataset,
                "dissemination": dissemination,
                "trials": len(group),
                "direct_edge_rate": group.direct_edge_FT.mean(),
                "reverse_path_rate": group.reverse_path_T_to_F.mean(),
                "same_scc_rate": group.same_scc_rate.mean(),
                "edge_without_reverse_path_rate": group.edge_without_reverse_path.mean(),
                "edge_with_reverse_path_rate": group.edge_with_reverse_path.mean(),
                "final_rate": group.final_F_before_T_rate.mean(),
                "final_given_edge_without_reverse_path": (
                    no_reverse.final_F_before_T_rate.mean()
                    if len(no_reverse) else float("nan")
                ),
                "final_given_edge_with_reverse_path": (
                    reverse.final_F_before_T_rate.mean()
                    if len(reverse) else float("nan")
                ),
            })
        pd.DataFrame(edge_rows).to_csv(
            os.path.join(args.output_dir, "edge_to_output_summary.csv"),
            index=False,
        )

        paired_rows = []
        join_keys = [
            "dataset", "deployment", "trial", "dissemination", "reporter_policy"
        ]
        for reference_arm in ["train_best_singleton", "oracle_best_singleton"]:
            reference = results[results.arm == reference_arm]
            paired = full_union.merge(
                reference,
                on=join_keys,
                suffixes=("_union", "_reference"),
            )
            paired["capacity_gain"] = (
                paired.total_honest_capacity_union
                - paired.total_honest_capacity_reference
            )
            paired["direct_edge_gain"] = (
                paired.direct_edge_FT_union - paired.direct_edge_FT_reference
            )
            paired["final_gain"] = (
                paired.final_F_before_T_rate_union
                - paired.final_F_before_T_rate_reference
            )
            for group_index, ((dataset, dissemination), group) in enumerate(
                paired.groupby(["dataset", "dissemination"], sort=True)
            ):
                row = {
                    "dataset": dataset,
                    "dissemination": dissemination,
                    "reference_arm": reference_arm,
                    "trials": len(group),
                    "deployments": group.deployment.nunique(),
                }
                for metric_index, metric in enumerate(
                    ["capacity_gain", "direct_edge_gain", "final_gain"]
                ):
                    row[f"{metric}_mean"] = group[metric].mean()
                    low, high = MATCHED.hierarchical_bootstrap(
                        group, metric,
                        args.seed + 920001 + group_index * 101 + metric_index,
                        args.bootstrap_repetitions,
                    )
                    row[f"{metric}_ci95_low"] = low
                    row[f"{metric}_ci95_high"] = high
                paired_rows.append(row)
        pd.DataFrame(paired_rows).to_csv(
            os.path.join(args.output_dir, "paired_policy_gain_summary.csv"),
            index=False,
        )

    overlaps = combined["winner_overlap.csv"]
    if not overlaps.empty:
        overlaps["both_empty"] = (overlaps["union"] == 0).astype(int)
        overlaps.loc[overlaps["both_empty"] == 1, "jaccard"] = float("nan")
        overlaps.to_csv(
            os.path.join(args.output_dir, "winner_overlap.csv"), index=False
        )
        overlaps.groupby("dataset").jaccard.describe().to_csv(
            os.path.join(args.output_dir, "winner_overlap_summary.csv")
        )
        MATCHED.plot_overlap(overlaps, args.output_dir)

    separations = combined["separation_trials.csv"]
    if not separations.empty:
        if "capacity_gain" not in separations:
            separations["capacity_gain"] = (
                separations.union_capacity - separations.max_singleton_capacity
            )
            separations["direct_edge_gain"] = (
                separations.union_direct_edge - separations.max_singleton_direct_edge
            )
            separations["final_rate_gain"] = (
                separations.union_final_rate - separations.max_singleton_final_rate
            )
            separations.to_csv(
                os.path.join(args.output_dir, "separation_trials.csv"), index=False
            )
        separation_summary = MATCHED.summarize_separation(separations, summary_args)
        separation_summary.to_csv(
            os.path.join(args.output_dir, "separation_summary.csv"), index=False
        )
        MATCHED.plot_separation(separation_summary, args.output_dir)

    conformance_dir = os.path.join(args.output_dir, "conformance")
    os.makedirs(conformance_dir, exist_ok=True)
    for shard in args.shards:
        source_dir = os.path.join(shard, "conformance")
        if not os.path.isdir(source_dir):
            continue
        for filename in os.listdir(source_dir):
            shutil.copy2(
                os.path.join(source_dir, filename),
                os.path.join(conformance_dir, filename),
            )

    shard_metadata = []
    for shard in args.shards:
        path = os.path.join(shard, "metadata.json")
        if os.path.exists(path):
            with open(path) as file:
                shard_metadata.append(json.load(file))
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as file:
        json.dump({
            "merged_shards": [os.path.abspath(path) for path in args.shards],
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "confidence_interval": (
                "95% percentile hierarchical bootstrap; resample deployments, "
                "then trials within deployments"
            ),
            "seed": args.seed,
            "rows": {name: len(frame) for name, frame in combined.items()},
            "shard_metadata": shard_metadata,
        }, file, indent=2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--bootstrap_repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
