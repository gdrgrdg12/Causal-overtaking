#!/usr/bin/env python3
"""Render focused publication figures from consolidated evaluation CSVs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"aws": "#1F6F8B", "wp": "#D97706"}
METRICS = {
    "strict_direct_edge_FT": ("Strict edge", "-", "o"),
    "final_F_before_T_rate": ("Final $F <_{out} T$", ":", "s"),
    "strict_edge_and_final_success": ("Strict edge and final", "--", "^"),
}


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
    })


def save(fig, root, stem):
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(root / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def percent_axis(ax, top=1.0):
    ax.set_ylim(0, top)
    ticks = np.linspace(0, top, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * value:.0f}%" for value in ticks])


def plot_scaling(root):
    data = pd.read_csv(root / "paper_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, dataset in zip(axes, ("aws", "wp")):
        subset = data[data.dataset == dataset].sort_values("n")
        x = np.arange(len(subset))
        for metric, (label, linestyle, marker) in METRICS.items():
            y = subset[metric].to_numpy()
            low = subset[f"{metric}_ci95_low"].to_numpy()
            high = subset[f"{metric}_ci95_high"].to_numpy()
            ax.errorbar(
                x, y, yerr=np.vstack((y - low, high - y)), label=label,
                color=COLORS[dataset], linestyle=linestyle, marker=marker,
                linewidth=2.0, markersize=5.5, capsize=3,
            )
        ax.set_title(f"{dataset.upper()} trace")
        ax.set_xticks(x, subset.n.astype(str))
        ax.set_xlabel("Committee size $n$")
        percent_axis(ax)
    axes[0].set_ylabel("Trial rate")
    axes[1].legend(frameon=False, loc="lower right")
    save(fig, root, "paper_scaling")


def plot_composition(root):
    data = pd.read_csv(root / "paper_composition.csv")
    arms = [
        "independent_singleton_truthful",
        "independent_singleton_reports",
        "five_observer_union_truthful",
        "five_observer_joint",
    ]
    labels = ["1 obs.\ntruthful", "1 obs.\n+ reports", "5 obs.\ntruthful", "5 obs.\n+ reports"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    width = 0.24
    metrics = [
        ("strict_edge", "Strict edge", "#436F8A"),
        ("final_F_before_T", "Final $F <_{out} T$", "#B5C6D1"),
        ("strict_edge_and_final", "Both", "#E58B2B"),
    ]
    for ax, dataset in zip(axes, ("aws", "wp")):
        indexed = data[data.dataset == dataset].set_index("arm").loc[arms]
        x = np.arange(len(arms))
        for offset, (metric, label, color) in enumerate(metrics):
            y = indexed[metric].to_numpy()
            low = indexed[f"{metric}_ci95_low"].to_numpy()
            high = indexed[f"{metric}_ci95_high"].to_numpy()
            ax.bar(
                x + (offset - 1) * width, y, width, label=label, color=color,
                edgecolor="#24333D", linewidth=0.6,
                yerr=np.vstack((y - low, high - y)), capsize=2.5,
                error_kw={"elinewidth": 0.9},
            )
        ax.set_title(f"{dataset.upper()} trace")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Attacker arm")
        percent_axis(ax)
    axes[0].set_ylabel("Trial rate")
    axes[1].legend(frameon=False, loc="upper left")
    save(fig, root, "paper_composition")


def plot_sensitivity(root):
    placement = pd.read_csv(root / "paper_placement.csv")
    rho = pd.read_csv(root / "paper_rho.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))

    placement_order = [
        "static-random", "region-medoid", "history-optimized-global"
    ]
    placement_labels = ["Static\nrandom", "Region\nmedoid", "History\noptimized"]
    x = np.arange(3)
    width = 0.34
    for offset, dataset in enumerate(("aws", "wp")):
        indexed = placement[placement.dataset == dataset].set_index(
            "placement"
        ).loc[placement_order]
        y = indexed.strict_edge_and_final_success.to_numpy()
        low = indexed.strict_edge_and_final_success_ci95_low.to_numpy()
        high = indexed.strict_edge_and_final_success_ci95_high.to_numpy()
        axes[0].bar(
            x + (offset - 0.5) * width, y, width, label=dataset.upper(),
            color=COLORS[dataset], yerr=np.vstack((y - low, high - y)),
            edgecolor="#24333D", linewidth=0.6, capsize=3,
        )
    axes[0].set_xticks(x, placement_labels)
    axes[0].set_title("Placement sensitivity")
    axes[0].set_ylabel("Strict edge and final")
    percent_axis(axes[0])
    axes[0].legend(frameon=False)

    for dataset in ("aws", "wp"):
        subset = rho[rho.dataset == dataset].sort_values("rho_ms")
        y = subset.strict_edge_and_final_success.to_numpy()
        low = subset.strict_edge_and_final_success_ci95_low.to_numpy()
        high = subset.strict_edge_and_final_success_ci95_high.to_numpy()
        axes[1].errorbar(
            subset.rho_ms, y, yerr=np.vstack((y - low, high - y)),
            label=dataset.upper(), color=COLORS[dataset], marker="o",
            linewidth=2, capsize=3,
        )
    axes[1].set_xticks([0, 1, 2, 5])
    axes[1].set_title("Reaction-delay sensitivity")
    axes[1].set_xlabel("Modeled reaction delay $\\rho$ (ms)")
    axes[1].set_ylabel("Strict edge and final")
    percent_axis(axes[1])
    axes[1].legend(frameon=False)
    save(fig, root, "paper_sensitivity")


def plot_guard(root):
    data = pd.read_csv(root / "paper_guard.csv")
    fig, axes = plt.subplots(
        2, 2, figsize=(10.4, 6.3),
        gridspec_kw={"width_ratios": [2.25, 1.0]},
    )
    attack_labels = [
        "Raw\nedge", "Carry\n$|$ edge", "All-trial\ncarry",
        "Raw\nfinal", "Residual\n(all trials)", "Residual\n$|$ carry",
    ]
    attack_metrics = [
        "raw_strict_edge", "carry_given_edge", "all_trial_carry_rate",
        "raw_final_F_before_T", "guarded_all_trial_residual",
        "residual_given_carry",
    ]
    benign_labels = ["Pair false\ncarry", "Batch-any\nfalse carry"]
    benign_metrics = [
        "benign_pair_false_carry", "benign_batch_any_false_carry",
    ]
    width = 0.34
    for row_index, bound_mode in enumerate(("same-hour", "previous-hour")):
        prefix = "In-window" if bound_mode == "same-hour" else "Frozen held-out"
        for column, (metrics, labels, title) in enumerate((
            (attack_metrics, attack_labels, "attack outcomes"),
            (benign_metrics, benign_labels, "benign impact"),
        )):
            ax = axes[row_index, column]
            x = np.arange(len(metrics))
            for offset, dataset in enumerate(("aws", "wp")):
                result = data[
                    (data.dataset == dataset)
                    & (data.bound_mode == bound_mode)
                ].iloc[0]
                y = np.asarray([result[metric] for metric in metrics], dtype=float)
                low = np.asarray([
                    result[f"{metric}_ci95_low"] for metric in metrics
                ], dtype=float)
                high = np.asarray([
                    result[f"{metric}_ci95_high"] for metric in metrics
                ], dtype=float)
                errors = np.vstack((y - low, high - y))
                errors[:, np.isnan(errors).any(axis=0)] = 0.0
                bars = ax.bar(
                    x + (offset - 0.5) * width, np.nan_to_num(y), width,
                    label=dataset.upper(), color=COLORS[dataset],
                    edgecolor="#24333D", linewidth=0.6,
                    yerr=errors, capsize=3,
                )
                if column == 1:
                    for bar, value in zip(bars, y):
                        ax.annotate(
                            f"{100 * value:.2f}%",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5,
                        )
            ax.set_title(f"{prefix}: {title}")
            ax.set_xticks(x, labels)
            percent_axis(ax, 1.0 if column == 0 else 0.8)
            if column == 0:
                ax.set_ylabel("Rate")
    axes[0, 1].legend(frameon=False, loc="upper left")
    save(fig, root, "paper_guard")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    setup()
    plot_scaling(root)
    plot_composition(root)
    plot_sensitivity(root)
    plot_guard(root)
    print(root)


if __name__ == "__main__":
    main()
