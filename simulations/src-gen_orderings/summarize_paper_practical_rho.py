#!/usr/bin/env python3
"""Summarize a frozen-placement rho sweep and its local benchmark."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("aws", "wp")
DISSEMINATIONS = ("plain", "delayed")
METRICS = (
    "strict_direct_edge_FT",
    "final_F_before_T_rate",
    "strict_edge_and_final_success",
)
TRIAL_MATCH_COLUMNS = (
    "dataset", "deployment", "trial", "reporter_policy", "replica_layout",
    "observation_model", "corrupted_observers", "corrupted_observer_ids",
    "corrupted_observer_locations", "corrupted_placement", "b",
    "byzantine_report_mode", "arm", "entry_count", "entry_ids",
    "entry_locations", "victim", "reporter_ids", "pair_processing_first",
    "public_repetitions",
)
MANIFEST_MATCH_COLUMNS = (
    "replicas", "clients", "entries", "reporter_order", "replica_layout",
    "replica_region_count", "replica_region_ids", "observation_model",
    "corrupted_observers", "corrupted_observer_ids",
    "corrupted_observer_locations", "corrupted_placement",
    "byzantine_report_mode",
)


def parse_values(value):
    return [float(item) for item in value.split()]


def scaling_dir(root, dataset):
    return root / "scaling" / dataset / "n21"


def rho_dir(root, dataset, rho):
    return root / "rho" / dataset / f"rho-{rho:g}"


def read_trials(directory, dissemination):
    frame = pd.read_csv(directory / "trial_results.csv")
    frame = frame[
        (frame.dissemination == dissemination)
        & (frame.arm == "full_union")
        & (frame.reporter_policy == "implementation")
    ].copy()
    if len(frame) != 400:
        raise ValueError(
            f"expected 400 {dissemination} trials in {directory}, found {len(frame)}"
        )
    frame["strict_edge_and_final_success"] = (
        frame.strict_direct_edge_FT * frame.final_F_before_T_rate
    )
    return frame.sort_values(["deployment", "trial"]).reset_index(drop=True)


def assert_columns_equal(left, right, columns, label):
    for column in columns:
        a = left[column].fillna("<NA>").astype(str).reset_index(drop=True)
        b = right[column].fillna("<NA>").astype(str).reset_index(drop=True)
        if not a.equals(b):
            mismatch = int(np.flatnonzero((a != b).to_numpy())[0])
            raise AssertionError(f"{label}: {column} differs at row {mismatch}")


def summary_row(frame, dataset, rho, dissemination, gossip_p, batch_size):
    result = {
        "dataset": dataset,
        "rho_ms": rho,
        "dissemination": dissemination,
        "gossip_p": 0.0 if dissemination == "plain" else gossip_p,
        "batch_size": 0 if dissemination == "plain" else batch_size,
        "deployments": int(frame.deployments.iloc[0]),
        "trials": int(frame.rows.iloc[0]),
    }
    for metric in METRICS:
        for suffix in ("mean", "ci95_low", "ci95_high"):
            output = metric if suffix == "mean" else f"{metric}_{suffix}"
            result[output] = float(frame[f"{metric}_{suffix}"].iloc[0])
    return result


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plain_root = Path(args.plain_root)
    gossip_root = Path(args.gossip_root)
    rho_root = Path(args.rho_root)
    rho_values = parse_values(args.rho_values)
    rows = []
    validation = []

    baselines = {}
    for dataset in DATASETS:
        plain_dir = scaling_dir(plain_root, dataset)
        gossip_dir = scaling_dir(gossip_root, dataset)
        baselines[(dataset, "plain")] = read_trials(plain_dir, "plain")
        baselines[(dataset, "delayed")] = read_trials(gossip_dir, "delayed")
        for dissemination, directory in (
            ("plain", plain_dir), ("delayed", gossip_dir)
        ):
            summary = pd.read_csv(directory / "summary.csv")
            summary = summary[
                (summary.dissemination == dissemination)
                & (summary.arm == "full_union")
                & (summary.reporter_policy == "implementation")
            ]
            rows.append(summary_row(
                summary, dataset, 0.0, dissemination,
                args.gossip_p, args.batch_size,
            ))

    for dataset in DATASETS:
        baseline_manifest = pd.read_csv(
            scaling_dir(plain_root, dataset) / "deployment_manifest.csv"
        ).sort_values("deployment").reset_index(drop=True)
        for rho in rho_values:
            directory = rho_dir(rho_root, dataset, rho)
            manifest = pd.read_csv(
                directory / "deployment_manifest.csv"
            ).sort_values("deployment").reset_index(drop=True)
            assert_columns_equal(
                baseline_manifest, manifest, MANIFEST_MATCH_COLUMNS,
                f"{dataset}/rho={rho:g} placement",
            )
            for dissemination in DISSEMINATIONS:
                trials = read_trials(directory, dissemination)
                assert_columns_equal(
                    baselines[(dataset, dissemination)], trials,
                    TRIAL_MATCH_COLUMNS,
                    f"{dataset}/{dissemination}/rho={rho:g} trial",
                )
                summary = pd.read_csv(directory / "summary.csv")
                summary = summary[
                    (summary.dissemination == dissemination)
                    & (summary.arm == "full_union")
                    & (summary.reporter_policy == "implementation")
                ]
                rows.append(summary_row(
                    summary, dataset, rho, dissemination,
                    args.gossip_p, args.batch_size,
                ))
            validation.append(
                f"{dataset}/rho={rho:g}: placement and 800 trial keys match rho=0"
            )

    result = pd.DataFrame(rows).sort_values(
        ["dataset", "dissemination", "rho_ms"]
    )
    result.to_csv(output / "paper_rho_frozen.csv", index=False)
    benchmark = pd.read_csv(args.rho_benchmark)
    release = benchmark[benchmark.metric == "rho_send_done_us"].copy()
    release.to_csv(output / "rho_local_measurement.csv", index=False)
    (output / "VALIDATION.md").write_text(
        "# Frozen-Placement Validation\n\n"
        + "\n".join(f"- {item}" for item in validation) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "claim": "frozen-placement reaction-delay sensitivity",
        "placement_rho_ms": 0.0,
        "execution_rho_ms": [0.0, *rho_values],
        "disseminations": ["plain", "delayed"],
        "gossip_p": args.gossip_p,
        "batch_size": args.batch_size,
        "local_measurement_release_point": "rho_send_done_us",
        "rho_benchmark": str(Path(args.rho_benchmark).resolve()),
    }
    (output / "rho_frozen_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_results(output, result, release, args)
    plot_results(output, result)


def interval(row, metric):
    return (
        f"{100 * row[metric]:.1f}% "
        f"[{100 * row[f'{metric}_ci95_low']:.1f}%, "
        f"{100 * row[f'{metric}_ci95_high']:.1f}%]"
    )


def write_results(output, result, release, args):
    lines = [
        "# Frozen-Placement Reaction-Delay Sensitivity",
        "",
        "All rho points reuse the rho=0 history-trained observer placement, "
        "deployment, victim, network sample, reporter set, background workload, "
        "gossip coins, and public Unspool tapes. Only the observation-to-release "
        "delay changes.",
        "",
        "| Trace | Dissemination | rho | Strict edge | Final F<T | Edge and final |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for values in result.to_dict("records"):
        lines.append(
            f"| {values['dataset'].upper()} | {values['dissemination']} | "
            f"{values['rho_ms']:g} ms | "
            f"{interval(values, 'strict_direct_edge_FT')} | "
            f"{interval(values, 'final_F_before_T_rate')} | "
            f"{interval(values, 'strict_edge_and_final_success')} |"
        )
    lines.extend([
        "",
        "## Local rho measurement",
        "",
        "| Mode | p50 | p95 | p99 | p99.9 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in release.itertuples(index=False):
        lines.append(
            f"| {row.mode} | {row.p50_us:.3f} us | {row.p95_us:.3f} us | "
            f"{row.p99_us:.3f} us | {row.p999_us:.3f} us |"
        )
    lines.extend([
        "",
        "The 0.1 ms point conservatively exceeds the measured signed p99.9 local "
        "release time. It is not an end-to-end public-feed or remote-provider "
        "measurement.",
        "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(output, result):
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    colors = {"strict_direct_edge_FT": "#2878A8", "final_F_before_T_rate": "#D66A1F"}
    labels = {"strict_direct_edge_FT": "Strict edge", "final_F_before_T_rate": "Final"}
    for ax, dataset in zip(axes, DATASETS):
        for dissemination in DISSEMINATIONS:
            subset = result[
                (result.dataset == dataset)
                & (result.dissemination == dissemination)
            ].sort_values("rho_ms")
            for metric in ("strict_direct_edge_FT", "final_F_before_T_rate"):
                y = subset[metric].to_numpy()
                low = subset[f"{metric}_ci95_low"].to_numpy()
                high = subset[f"{metric}_ci95_high"].to_numpy()
                ax.errorbar(
                    subset.rho_ms, y, yerr=np.vstack((y - low, high - y)),
                    color=colors[metric],
                    linestyle="-" if dissemination == "plain" else "--",
                    marker="o" if metric == "strict_direct_edge_FT" else "s",
                    linewidth=2, capsize=3,
                    label=f"{labels[metric]}, {dissemination}",
                )
        ax.set_title(dataset.upper())
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xticks([0, 0.1, 1, 2, 5], ["0", "0.1", "1", "2", "5"])
        ax.set_xlabel("Reaction delay $\\rho$ (ms)")
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.linspace(0, 1, 5), ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Trials")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"paper_rho_frozen.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_root", required=True)
    parser.add_argument("--gossip_root", required=True)
    parser.add_argument("--rho_root", required=True)
    parser.add_argument("--rho_benchmark", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rho_values", default="0.1 1 2 5")
    parser.add_argument("--gossip_p", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
