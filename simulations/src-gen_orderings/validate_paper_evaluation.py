#!/usr/bin/env python3
"""Fail closed when the frozen paper artifact is incomplete or inconsistent."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_trial_keys(path, expected_rows, expected_deployments):
    frame = pd.read_csv(path)
    require(len(frame) == expected_rows, f"{path}: expected {expected_rows} rows")
    require(
        frame.deployment.nunique() == expected_deployments,
        f"{path}: wrong deployment count",
    )
    require(
        not frame.duplicated(["deployment", "trial"]).any(),
        f"{path}: duplicate deployment/trial keys",
    )
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--expected_deployments", type=int, default=20)
    parser.add_argument("--expected_trials", type=int, default=20)
    parser.add_argument("--expected_pair_trials", type=int, default=100)
    args = parser.parse_args()
    root = args.result_root.resolve()
    checks = []
    full_rows = args.expected_deployments * args.expected_trials
    pair_rows = args.expected_deployments * args.expected_pair_trials

    for dataset in ("aws", "wp"):
        for n in (9, 21, 101):
            path = root / "scaling" / dataset / f"n{n}" / "trial_results.csv"
            scaling_trials = validate_trial_keys(
                path, full_rows, args.expected_deployments
            )
            require(
                (scaling_trials.unique_region_count <=
                 scaling_trials.attack_source_count).all(),
                f"{path}: physical regions exceed source identities",
            )
            require(
                not scaling_trials.entry_locations.astype(str).str.contains(
                    "@@replica=", regex=False
                ).any(),
                f"{path}: logical aliases leaked into physical regions",
            )
            checks.append(
                f"scaling/{dataset}/n{n}: {full_rows} unique trials"
            )

        composition = root / "composition" / dataset
        trials = pd.read_csv(composition / "trial_results.csv")
        require(len(trials) == pair_rows, f"{dataset} composition trial count")
        require(
            trials.deployment.nunique() == args.expected_deployments,
            f"{dataset} composition deployments",
        )
        arms = pd.read_csv(composition / "arm_results.csv")
        require(len(arms) == pair_rows * 5, f"{dataset} composition arm rows")
        require(set(arms.arm.unique()) == {
            "history-singleton+honest", "history-singleton+reports",
            "any-singleton+reports", "observer-union+honest", "joint",
        }, f"{dataset} composition arms")
        checks.append(
            f"composition/{dataset}: {pair_rows} paired traces and five arms"
        )

        for placement in (
            "static-random", "region-medoid", "history-optimized-global"
        ):
            validate_trial_keys(
                root / "placement" / dataset / placement / "trial_results.csv",
                full_rows, args.expected_deployments,
            )
        checks.append(f"placement/{dataset}: three paired placements")

        for rho in (1, 2, 5):
            validate_trial_keys(
                root / "rho" / dataset / f"rho-{rho}" / "trial_results.csv",
                full_rows, args.expected_deployments,
            )
        checks.append(f"rho/{dataset}: 0 ms baseline plus 1/2/5 ms")

        for bound in ("same-hour", "previous-hour"):
            guard = root / "guard-matched" / dataset / bound
            validate_trial_keys(
                guard / "trial_summary.csv", full_rows,
                args.expected_deployments,
            )
            components = pd.read_csv(guard / "component_summary.csv")
            require(len(components) == 20, f"{guard}: component table incomplete")
            require(set(components.policy.unique()) == {
                "no-guard", "profile-only", "capacity-only",
                "profile+trace-bound", "carry-every-reverse-edge",
            }, f"{guard}: component policies incomplete")
            conformance = pd.read_csv(guard / "original_unspool_conformance.csv")
            require(
                len(conformance) == args.expected_deployments,
                f"{guard}: wrong C++ case count",
            )
            require(conformance.exact_match.eq(1).all(), f"{guard}: C++ mismatch")
        checks.append(
            f"guard-matched/{dataset}: {2 * full_rows} workloads and "
            f"{2 * args.expected_deployments} exact C++ replays"
        )

    budget_dir = root / "corruption-budget"
    budget = pd.read_csv(
        budget_dir / "trial_results.csv.gz", low_memory=False
    )
    expected_budget_rows = 2 * 5 * pair_rows * 4
    require(
        len(budget) == expected_budget_rows,
        "corruption-budget trial count",
    )
    require(
        not budget.duplicated([
            "dataset", "deployment", "actual_corruptions", "trial",
            "strategy",
        ]).any(),
        "corruption-budget duplicate trial keys",
    )
    require(
        set(budget.strategy.unique()) == {
            "observers_truthful", "single_observer_reports",
            "observers_one_report", "joint",
        },
        "corruption-budget strategies",
    )
    require(
        budget.required_corruptions.eq(budget.actual_corruptions).all(),
        "corruption-budget max(k,b)=c invariant",
    )
    alias_columns = [
        "w_FT", "w_TF", "strict_edge", "artifact_direct_edge_FT",
        "artifact_direct_edge_TF", "same_scc_rate",
        "final_F_before_T_rate", "strict_edge_and_final_rate",
    ]
    aliases = budget[
        (budget.actual_corruptions == 1)
        & (budget.strategy != "observers_truthful")
    ]
    for _, group in aliases.groupby(["dataset", "deployment", "trial"]):
        for column in alias_columns:
            values = group[column].dropna().to_numpy(dtype=float)
            require(
                not len(values) or np.allclose(
                    values, values[0], rtol=0, atol=1e-12
                ),
                f"corruption-budget c=1 alias mismatch: {column}",
            )
    full_budget = budget[budget.final_F_before_T_rate.notna()]
    require(
        not ((full_budget.strict_edge == 1)
             & (full_budget.artifact_direct_edge_FT != 1)).any(),
        "corruption-budget full graph dropped a strict edge",
    )
    require(
        len(full_budget) == 2 * 5 * full_rows * 4,
        "corruption-budget final workload count",
    )

    legacy_map = {
        "observers_truthful": (
            "observer-union+honest", "observer_union_honest"
        ),
        "single_observer_reports": (
            "history-singleton+reports", "history_singleton_reports"
        ),
        "joint": ("joint", "joint"),
    }
    for dataset in ("aws", "wp"):
        current = budget[
            (budget.dataset == dataset)
            & (budget.actual_corruptions == 5)
        ]
        legacy_dir = root / "composition" / dataset
        legacy_arms = pd.read_csv(legacy_dir / "arm_results.csv")
        legacy_trials = pd.read_csv(legacy_dir / "trial_results.csv")
        for strategy, (legacy_arm, legacy_prefix) in legacy_map.items():
            left = current[current.strategy == strategy].sort_values(
                ["deployment", "trial"]
            )
            right = legacy_arms[
                legacy_arms.arm == legacy_arm
            ].sort_values(["deployment", "trial"])
            for column in (
                "w_FT", "w_TF", "selected_honest_support", "strict_edge"
            ):
                require(
                    left[column].tolist() == right[column].tolist(),
                    f"{dataset} c=5 legacy pair mismatch: "
                    f"{strategy}/{column}",
                )
            left_final = left[left.trial < args.expected_trials]
            right_final = legacy_trials[
                legacy_trials.trial < args.expected_trials
            ].sort_values(["deployment", "trial"])
            for column in (
                "artifact_direct_edge_FT", "artifact_direct_edge_TF",
                "same_scc_rate", "final_F_before_T_rate",
                "strict_edge_and_final_rate",
            ):
                require(
                    np.allclose(
                        left_final[column],
                        right_final[f"{legacy_prefix}_{column}"],
                        rtol=0, atol=0,
                    ),
                    f"{dataset} c=5 legacy final mismatch: "
                    f"{strategy}/{column}",
                )
    for filename in (
        "summary.csv", "composition_gain.csv", "RESULTS.md",
        "metadata.json", "corruption_budget_ablation.pdf",
    ):
        require(
            (budget_dir / filename).is_file(),
            f"corruption-budget missing {filename}",
        )
    checks.append(
        "corruption-budget: c=1..5, four arms, max(k,b)=c, and exact c=5 "
        "legacy endpoint replay"
    )

    match = pd.read_csv(root / "paper_guard_section4_match.csv")
    require(len(match) == 4, "Guard/Section IV match audit is incomplete")
    require(
        match.all_fields_exact_matches.eq(match.paired_trials).all(),
        "matched Guard baseline differs from Section IV",
    )
    checks.append("Guard baseline exactly matches Section IV in all audited fields")

    consolidated = [
        "paper_scaling.csv", "paper_composition.csv",
        "paper_composition_separation.csv", "paper_placement.csv",
        "paper_rho.csv", "paper_guard.csv", "paper_guard_ablation.csv",
        "paper_guard_section4_match.csv", "RESULTS.md",
        "GUARD_MATCHED_REEVALUATION.md", "MANUSCRIPT_UPDATE_NOTES.md",
    ]
    for filename in consolidated:
        require((root / filename).is_file(), f"missing {filename}")
    checks.append("all consolidated tables and report exist")

    result = {
        "status": "pass",
        "checks": checks,
        "confidence_interval": (
            "nominal 95% percentile hierarchical bootstrap; deployment then trial"
        ),
        "seed": 762031,
    }
    (root / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Artifact Validation", "", "Status: **PASS**", ""]
    lines.extend(f"- {check}" for check in checks)
    (root / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "VALIDATION.md")


if __name__ == "__main__":
    main()
