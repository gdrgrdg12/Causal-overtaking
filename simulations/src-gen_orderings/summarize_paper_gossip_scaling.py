#!/usr/bin/env python3
"""Compare Figure 1 plain dissemination with paired delayed gossip."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matched_multientry_evaluation as matched


DATASETS = ("aws", "wp")
SCALES = (9, 21, 101)
METRICS = (
    "post_dissemination_honest_capacity",
    "strict_direct_edge_FT",
    "final_F_before_T_rate",
    "strict_edge_and_final_success",
)
MATCH_COLUMNS = (
    "dataset",
    "deployment",
    "trial",
    "reporter_policy",
    "replica_layout",
    "observation_model",
    "corrupted_observers",
    "corrupted_observer_ids",
    "corrupted_observer_locations",
    "corrupted_placement",
    "b",
    "byzantine_report_mode",
    "rho_ms",
    "arm",
    "entry_count",
    "entry_ids",
    "entry_locations",
    "victim",
    "reporter_ids",
    "pair_processing_first",
    "public_repetitions",
)


def case_dir(root, dataset, n):
    return root / "scaling" / dataset / f"n{n}"


def read_trials(root, dataset, n, dissemination):
    path = case_dir(root, dataset, n) / "trial_results.csv"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["dissemination"] == dissemination)
        & (frame["arm"] == "full_union")
        & (frame["reporter_policy"] == "implementation")
    ].copy()
    if len(frame) != 400:
        raise ValueError(f"expected 400 rows in {path}, found {len(frame)}")
    frame["strict_edge_and_final_success"] = (
        frame["strict_direct_edge_FT"] * frame["final_F_before_T_rate"]
    )
    return frame.sort_values(["deployment", "trial"]).reset_index(drop=True)


def assert_paired(plain, delayed, dataset, n):
    for column in MATCH_COLUMNS:
        left = plain[column].fillna("<NA>").astype(str)
        right = delayed[column].fillna("<NA>").astype(str)
        if not left.equals(right):
            mismatch = int(np.flatnonzero((left != right).to_numpy())[0])
            raise AssertionError(
                f"{dataset}/n{n} is not paired in {column} at row {mismatch}"
            )


def bootstrap(frame, metric, seed, repetitions):
    return matched.hierarchical_bootstrap(frame, metric, seed, repetitions)


def summarize(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plain_root = Path(args.plain_root)
    delayed_root = Path(args.delayed_root)
    rows = []
    paired_rows = []
    validation = []
    for dataset_index, dataset in enumerate(DATASETS):
        for scale_index, n in enumerate(SCALES):
            plain = read_trials(plain_root, dataset, n, "plain")
            delayed = read_trials(delayed_root, dataset, n, "delayed")
            assert_paired(plain, delayed, dataset, n)
            validation.append(f"{dataset}/n{n}: 400/400 counterfactual keys match")
            for dissemination_index, (name, frame) in enumerate((
                ("plain", plain), ("delayed", delayed)
            )):
                row = {
                    "dataset": dataset,
                    "n": n,
                    "dissemination": name,
                    "gossip_p": 0.0 if name == "plain" else args.gossip_p,
                    "batch_size": 0 if name == "plain" else args.batch_size,
                    "deployments": frame.deployment.nunique(),
                    "trials": len(frame),
                }
                for metric_index, metric in enumerate(METRICS):
                    row[metric] = float(frame[metric].mean())
                    low, high = bootstrap(
                        frame, metric,
                        args.seed + dataset_index * 10000 + scale_index * 1000
                        + dissemination_index * 100 + metric_index,
                        args.bootstrap_repetitions,
                    )
                    row[f"{metric}_ci95_low"] = low
                    row[f"{metric}_ci95_high"] = high
                rows.append(row)

            paired = plain[["deployment", "trial"]].copy()
            paired["dataset"] = dataset
            paired["n"] = n
            paired_row = {
                "dataset": dataset,
                "n": n,
                "deployments": paired.deployment.nunique(),
                "trials": len(paired),
            }
            for metric_index, metric in enumerate(METRICS):
                delta = f"delta_{metric}"
                paired[delta] = delayed[metric].to_numpy() - plain[metric].to_numpy()
                paired_row[delta] = float(paired[delta].mean())
                low, high = bootstrap(
                    paired, delta,
                    args.seed + 50000 + dataset_index * 10000
                    + scale_index * 1000 + metric_index,
                    args.bootstrap_repetitions,
                )
                paired_row[f"{delta}_ci95_low"] = low
                paired_row[f"{delta}_ci95_high"] = high
            paired_rows.append(paired_row)

    summary = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    summary.to_csv(output / "paper_gossip_scaling.csv", index=False)
    paired.to_csv(output / "paper_gossip_scaling_paired.csv", index=False)
    (output / "VALIDATION.md").write_text(
        "# Paired Validation\n\n" + "\n".join(f"- {line}" for line in validation)
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "claim": "paired Figure 1 dissemination sensitivity",
        "plain_root": str(plain_root.resolve()),
        "delayed_root": str(delayed_root.resolve()),
        "gossip_p": args.gossip_p,
        "batch_size": args.batch_size,
        "gossip_rng_scheme": "trial-seeded-pcg64-v1",
        "gossip_latency_sampling": "stable-keyed sample at each actual flush",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "confidence_interval": (
            "95% percentile hierarchical bootstrap; resample deployments, "
            "then trials within deployments"
        ),
        "seed": args.seed,
        "paired_columns": list(MATCH_COLUMNS),
    }
    (output / "gossip_scaling_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_results(output, summary, paired, args)
    plot_results(output, summary)


def percent(value):
    return f"{100 * value:.1f}%"


def interval(row, metric):
    return (
        f"{percent(row[metric])} "
        f"[{percent(row[f'{metric}_ci95_low'])}, "
        f"{percent(row[f'{metric}_ci95_high'])}]"
    )


def write_results(output, summary, paired, args):
    lines = [
        "# Figure 1 Delayed-Gossip Sensitivity",
        "",
        "This experiment replays the six Figure 1 configurations with the same "
        "deployments, victims, execution samples, reporter sets, pair-processing "
        "orders, background workloads, and public Unspool tapes. The only changed "
        f"factor is delayed gossip (`p={args.gossip_p}`, batch size "
        f"`{args.batch_size}`).",
        "",
        "| Trace | n | Dissemination | Strict edge | Final F<T | Edge and final |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        data = row._asdict()
        lines.append(
            f"| {row.dataset.upper()} | {row.n} | {row.dissemination} | "
            f"{interval(data, 'strict_direct_edge_FT')} | "
            f"{interval(data, 'final_F_before_T_rate')} | "
            f"{interval(data, 'strict_edge_and_final_success')} |"
        )
    lines.extend([
        "",
        "## Paired delayed-minus-plain change",
        "",
        "| Trace | n | Strict edge | Final F<T | Edge and final |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in paired.itertuples(index=False):
        data = row._asdict()
        values = []
        for metric in (
            "strict_direct_edge_FT", "final_F_before_T_rate",
            "strict_edge_and_final_success",
        ):
            name = f"delta_{metric}"
            values.append(
                f"{100 * data[name]:+.1f} pp "
                f"[{100 * data[f'{name}_ci95_low']:+.1f}, "
                f"{100 * data[f'{name}_ci95_high']:+.1f}]"
            )
        lines.append(
            f"| {row.dataset.upper()} | {row.n} | " + " | ".join(values) + " |"
        )
    lines.extend([
        "",
        "Delayed gossip is reported as a propagation-policy sensitivity, not as "
        "the attack's default model and not automatically as a mitigation claim.",
        "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(output, summary):
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    styles = {
        ("strict_direct_edge_FT", "plain"): ("#2878A8", "-", "o", "Strict edge, plain"),
        ("strict_direct_edge_FT", "delayed"): ("#2878A8", "--", "o", "Strict edge, gossip"),
        ("final_F_before_T_rate", "plain"): ("#D66A1F", "-", "s", "Final, plain"),
        ("final_F_before_T_rate", "delayed"): ("#D66A1F", "--", "s", "Final, gossip"),
    }
    for ax, dataset in zip(axes, DATASETS):
        for (metric, dissemination), (color, line, marker, label) in styles.items():
            subset = summary[
                (summary.dataset == dataset)
                & (summary.dissemination == dissemination)
            ].sort_values("n")
            y = subset[metric].to_numpy()
            low = subset[f"{metric}_ci95_low"].to_numpy()
            high = subset[f"{metric}_ci95_high"].to_numpy()
            ax.errorbar(
                np.arange(3), y, yerr=np.vstack((y - low, high - y)),
                color=color, linestyle=line, marker=marker, linewidth=2,
                markersize=5.5, capsize=3, label=label,
            )
        ax.set_title(dataset.upper())
        ax.set_xticks(np.arange(3), ["9\n($f=2$)", "21\n($f=5$)", "101\n($f=25$)"])
        ax.set_xlabel("Committee size $n$")
        ax.set_ylim(0, 1.03)
        ax.set_yticks(np.linspace(0, 1, 5), ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Trials")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"paper_gossip_scaling.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain_root", required=True)
    parser.add_argument("--delayed_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gossip_p", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=762031)
    return parser


if __name__ == "__main__":
    summarize(build_parser().parse_args())
