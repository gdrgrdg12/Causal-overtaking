#!/usr/bin/env python3
"""Consolidate the frozen paper-evaluation suite into publication tables."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("aws", "wp")
SCALES = (9, 21, 101)
PLACEMENTS = ("static-random", "region-medoid", "history-optimized-global")
RHO_VALUES = (0, 1, 2, 5)
BOOTSTRAP_REPETITIONS = 10000


def read_one(path):
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected one row in {path}, found {len(frame)}")
    return frame.iloc[0]


def metric_triplet(row, prefix):
    return {
        prefix: float(row[f"{prefix}_mean"]),
        f"{prefix}_ci95_low": float(row[f"{prefix}_ci95_low"]),
        f"{prefix}_ci95_high": float(row[f"{prefix}_ci95_high"]),
    }


def matched_row(path, dataset, **labels):
    row = read_one(path / "summary.csv")
    result = {"dataset": dataset, **labels}
    for metric in (
        "unique_region_count",
        "selected_honest_capacity",
        "strict_direct_edge_FT",
        "final_F_before_T_rate",
        "strict_edge_and_final_success",
    ):
        result.update(metric_triplet(row, metric))
    result["deployments"] = int(row["deployments"])
    result["trials"] = int(row["rows"])
    return result


def collect_scaling(root):
    rows = []
    for dataset in DATASETS:
        for n in SCALES:
            rows.append(matched_row(
                root / "scaling" / dataset / f"n{n}", dataset, n=n
            ))
    return pd.DataFrame(rows)


def collect_placement(root):
    rows = []
    for dataset in DATASETS:
        for placement in PLACEMENTS:
            rows.append(matched_row(
                root / "placement" / dataset / placement,
                dataset,
                placement=placement,
            ))
    return pd.DataFrame(rows)


def collect_rho(root):
    rows = []
    for dataset in DATASETS:
        for rho in RHO_VALUES:
            source = (
                root / "scaling" / dataset / "n21"
                if rho == 0
                else root / "rho" / dataset / f"rho-{rho}"
            )
            rows.append(matched_row(source, dataset, rho_ms=rho))
    return pd.DataFrame(rows)


def metric_lookup(frame, metric):
    rows = frame[frame["metric"] == metric]
    if len(rows) != 1:
        raise ValueError(f"expected one {metric!r} row, found {len(rows)}")
    row = rows.iloc[0]
    return float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"])


def collect_composition(root):
    arm_names = {
        "history-singleton+honest": "independent_singleton_truthful",
        "history-singleton+reports": "independent_singleton_reports",
        "any-singleton+reports": "oracle_singleton_reports_diagnostic",
        "observer-union+honest": "five_observer_union_truthful",
        "joint": "five_observer_joint",
    }
    final_prefixes = {
        "independent_singleton_truthful": "history_singleton_honest",
        "independent_singleton_reports": "history_singleton_reports",
        "oracle_singleton_reports_diagnostic": "oracle_singleton_reports",
        "five_observer_union_truthful": "observer_union_honest",
        "five_observer_joint": "joint",
    }
    rows = []
    separation_rows = []
    for dataset in DATASETS:
        directory = root / "composition" / dataset
        arms = pd.read_csv(directory / "arm_summary.csv")
        metrics = pd.read_csv(directory / "composition_summary.csv")
        for raw_name, output_name in arm_names.items():
            arm = arms[arms["arm"] == raw_name].iloc[0]
            prefix = final_prefixes[output_name]
            final = metric_lookup(metrics, f"{prefix}_final_F_before_T_rate")
            conjunction = metric_lookup(
                metrics, f"{prefix}_strict_edge_and_final_rate"
            )
            rows.append({
                "dataset": dataset,
                "arm": output_name,
                "strict_edge": float(arm["strict_edge_mean"]),
                "strict_edge_ci95_low": float(arm["strict_edge_ci95_low"]),
                "strict_edge_ci95_high": float(arm["strict_edge_ci95_high"]),
                "final_F_before_T": final[0],
                "final_F_before_T_ci95_low": final[1],
                "final_F_before_T_ci95_high": final[2],
                "strict_edge_and_final": conjunction[0],
                "strict_edge_and_final_ci95_low": conjunction[1],
                "strict_edge_and_final_ci95_high": conjunction[2],
                "pair_trials": int(arm["trials"]),
                "deployments": int(arm["deployments"]),
            })
        for label, metric in (
            ("reports_singleton_union", "strict_edge_separation"),
            ("truthful_singleton_union", "truthful_strict_edge_separation"),
            ("reports_singleton_union_and_final", "strict_separation_and_final_rate"),
            (
                "truthful_singleton_union_and_final",
                "truthful_strict_separation_and_final_rate",
            ),
        ):
            value = metric_lookup(metrics, metric)
            separation_rows.append({
                "dataset": dataset,
                "separation": label,
                "value": value[0],
                "ci95_low": value[1],
                "ci95_high": value[2],
            })
    return pd.DataFrame(rows), pd.DataFrame(separation_rows)


def summary_metric(frame, metric):
    rows = frame[frame["metric"] == metric]
    if len(rows) != 1:
        raise ValueError(f"expected one guard metric {metric!r}, found {len(rows)}")
    return rows.iloc[0]


def hierarchical_ratio(frame, numerator, denominator, seed):
    deployments = sorted(frame.deployment.unique())
    groups = [frame[frame.deployment == deployment] for deployment in deployments]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        selected = rng.integers(0, len(groups), len(groups))
        total_numerator = 0.0
        total_denominator = 0.0
        for group_index in selected:
            group = groups[group_index]
            sampled = rng.integers(0, len(group), len(group))
            total_numerator += group[numerator].to_numpy(dtype=float)[sampled].sum()
            total_denominator += group[denominator].to_numpy(dtype=float)[sampled].sum()
        if total_denominator:
            values.append(total_numerator / total_denominator)
    if not values:
        return math.nan, math.nan
    return tuple(np.quantile(values, [0.025, 0.975]))


def collect_guard(root, directory_name="guard-matched"):
    main_rows = []
    component_rows = []
    for dataset in DATASETS:
        for bound_mode in ("same-hour", "previous-hour"):
            directory = root / directory_name / dataset / bound_mode
            summary = pd.read_csv(directory / "summary.csv")
            components = pd.read_csv(directory / "component_summary.csv")
            attack = summary[
                (summary["scenario"] == "attack-fastest")
                & (summary["behavior"] == "pair-swap")
            ]
            benign = summary[
                (summary["scenario"] == "benign-background")
                & (summary["behavior"] == "honest")
            ]
            attack_trials = pd.read_csv(directory / "trial_summary.csv")
            raw_edge = attack_trials["attack_raw_edge"].astype(bool)
            carried = attack_trials["attack_deferred"].astype(bool)
            residual = attack_trials[
                "cross_epoch_final_F_before_T_rate"
            ].astype(float)
            residual_given_carry = (
                float(residual[carried].mean()) if carried.any() else math.nan
            )
            bootstrap_frame = attack_trials.assign(
                all_trials=1,
                carried=carried.astype(int),
                carried_residual=residual * carried.astype(int),
            )
            seed_offset = 10000 if dataset == "wp" else 0
            seed_offset += 1000 if bound_mode == "previous-hour" else 0
            carry_ci = hierarchical_ratio(
                bootstrap_frame, "carried", "all_trials", 762031 + seed_offset
            )
            residual_carry_ci = hierarchical_ratio(
                bootstrap_frame,
                "carried_residual",
                "carried",
                762032 + seed_offset,
            )
            row = {"dataset": dataset, "bound_mode": bound_mode}
            for output, metric, frame in (
                ("raw_strict_edge", "raw_strict_edge_over_all_trials", attack),
                ("carry_given_edge", "detected_given_raw_edge", attack),
                ("raw_final_F_before_T", "raw_final_F_before_T_rate", attack),
                (
                    "guarded_all_trial_residual",
                    "cross_epoch_final_F_before_T_rate",
                    attack,
                ),
                (
                    "benign_pair_false_carry",
                    "false_deferred_over_all_background_pairs",
                    benign,
                ),
                (
                    "benign_batch_any_false_carry",
                    "batch_has_any_false_deferred",
                    benign,
                ),
            ):
                source = summary_metric(frame, metric)
                row[output] = float(source["value"])
                row[f"{output}_ci95_low"] = float(source["ci95_low"])
                row[f"{output}_ci95_high"] = float(source["ci95_high"])
                row[f"{output}_numerator"] = int(source["numerator"])
                row[f"{output}_denominator"] = int(source["denominator"])
            row["all_trial_carry_rate"] = float(carried.mean())
            row["all_trial_carry_rate_ci95_low"] = carry_ci[0]
            row["all_trial_carry_rate_ci95_high"] = carry_ci[1]
            row["residual_given_carry"] = residual_given_carry
            row["residual_given_carry_ci95_low"] = residual_carry_ci[0]
            row["residual_given_carry_ci95_high"] = residual_carry_ci[1]
            row["attack_trials"] = len(attack_trials)
            main_rows.append(row)

            for _, source in components.iterrows():
                component_rows.append({
                    "dataset": dataset,
                    "bound_mode": bound_mode,
                    "policy": source["policy"],
                    "metric": source["metric"],
                    "value": source["value"],
                    "ci95_low": source["ci95_low"],
                    "ci95_high": source["ci95_high"],
                    "numerator": source["numerator"],
                    "denominator": source["denominator"],
                })
    return pd.DataFrame(main_rows), pd.DataFrame(component_rows)


def collect_guard_section4_match(root):
    rows = []
    for dataset in DATASETS:
        section4 = pd.read_csv(
            root / "scaling" / dataset / "n21" / "trial_results.csv"
        )
        keys = ["deployment", "trial"]
        for bound_mode in ("same-hour", "previous-hour"):
            guard = pd.read_csv(
                root / "guard-matched" / dataset / bound_mode
                / "trial_summary.csv"
            )
            merged = section4.merge(
                guard, on=keys, suffixes=("_section4", "_guard"),
                validate="one_to_one",
            )
            checks = {
                "victim": (
                    merged.victim_section4.astype(str)
                    == merged.victim_guard.astype(str)
                ),
                "strict_edge": (
                    merged.strict_direct_edge_FT.astype(int)
                    == merged.attack_raw_edge.astype(int)
                ),
                "direct_edge": (
                    merged.direct_edge_FT.astype(int)
                    == merged.raw_direct_edge_FT.astype(int)
                ),
                "w_FT": (
                    merged.w_FT.astype(int)
                    == merged.attack_w_FT.astype(int)
                ),
                "w_TF": (
                    merged.w_TF.astype(int)
                    == merged.attack_w_TF.astype(int)
                ),
                "final_rate": np.isclose(
                    merged.final_F_before_T_rate,
                    merged.raw_final_F_before_T_rate,
                ),
            }
            row = {
                "dataset": dataset,
                "bound_mode": bound_mode,
                "paired_trials": len(merged),
            }
            for name, values in checks.items():
                row[f"{name}_exact_matches"] = int(np.sum(values))
                row[f"{name}_match_rate"] = float(np.mean(values))
            row["all_fields_exact_matches"] = int(np.sum(np.logical_and.reduce(
                [np.asarray(values) for values in checks.values()]
            )))
            row["all_fields_match_rate"] = (
                row["all_fields_exact_matches"] / len(merged)
            )
            rows.append(row)
    return pd.DataFrame(rows)


def align_guard_raw_intervals(guard, scaling):
    """Use one frozen bootstrap tape for raw metrics shared with Section IV."""
    result = guard.copy()
    mappings = {
        "raw_strict_edge": "strict_direct_edge_FT",
        "raw_final_F_before_T": "final_F_before_T_rate",
    }
    for dataset in DATASETS:
        baseline = scaling[
            (scaling.dataset == dataset) & (scaling.n == 21)
        ].iloc[0]
        selected = result.dataset == dataset
        for guard_metric, scaling_metric in mappings.items():
            if not np.allclose(
                result.loc[selected, guard_metric], baseline[scaling_metric]
            ):
                raise ValueError(
                    f"{dataset} Guard {guard_metric} differs from Section IV"
                )
            result.loc[selected, guard_metric] = baseline[scaling_metric]
            for suffix in ("ci95_low", "ci95_high"):
                result.loc[selected, f"{guard_metric}_{suffix}"] = baseline[
                    f"{scaling_metric}_{suffix}"
                ]
    return result


def percent(value):
    if pd.isna(value):
        return "n/a"
    percentage = 100.0 * value
    if percentage == 0:
        return "0.0%"
    if abs(percentage) < 1:
        return f"{percentage:.3f}%"
    return f"{percentage:.1f}%"


def ci(value, low, high):
    return f"{percent(value)} [{percent(low)}, {percent(high)}]"


def markdown_table(frame, columns, headers=None):
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(root, scaling, composition, separation, placement, rho, guard, components):
    lines = [
        "# Frozen Paper Evaluation Results",
        "",
        "All intervals below are nominal 95% percentile hierarchical-bootstrap intervals: deployments are resampled first, then trials within each deployment.",
        "",
        "## Frozen experimental design",
        "",
        "- Each full-workload configuration contains 20 fixed deployments and 20 matched trials per deployment (400 trials). The composition support experiment uses 100 pair traces per deployment (2,000 traces) and retains 20 full Unspool workloads per deployment for final-order outcomes.",
        "- Every full workload contains T, F, and 300 background transactions generated by a 15-client Poisson process over a 1,000 ms epoch. T is released at 500 ms.",
        "- The protocol consumes n-f reports in a frozen implementation order. The legal worst-case execution includes f corrupted reporters using pair-swap reports; all honest report contents and all background-transaction positions remain unchanged.",
        "- Corrupted observer placement is trained only on history samples and then frozen. The ex-post best singleton appears only as an explicitly labeled diagnostic arm.",
        "- Trace RTT is divided by two as an explicit one-way-delay model. AWS provides 21 physical latency profiles; n=101 therefore uses distinct logical replica identities that may share those profiles, without treating co-location as independent path diversity.",
        "- Attack results use plain first-copy dissemination. Placement is a P1 sensitivity result; the headline BFT claim concerns a worst legal execution, not average placement or reporter selection.",
        "",
        "## Cross-scale attack",
        "",
    ]
    table = scaling.copy()
    table["unique_region_count"] = table.unique_region_count.round().astype(int)
    table["strict edge"] = table.apply(lambda r: ci(
        r.strict_direct_edge_FT, r.strict_direct_edge_FT_ci95_low,
        r.strict_direct_edge_FT_ci95_high), axis=1)
    table["final F<T"] = table.apply(lambda r: ci(
        r.final_F_before_T_rate, r.final_F_before_T_rate_ci95_low,
        r.final_F_before_T_rate_ci95_high), axis=1)
    table["edge and final"] = table.apply(lambda r: ci(
        r.strict_edge_and_final_success,
        r.strict_edge_and_final_success_ci95_low,
        r.strict_edge_and_final_success_ci95_high), axis=1)
    lines.extend([
        markdown_table(
            table,
            ["dataset", "n", "unique_region_count", "strict edge", "final F<T", "edge and final"],
            ["dataset", "n", "physical source regions", "strict edge", "final F<T", "edge and final"],
        ),
        "",
        (
            "The strict-edge-and-final conjunction rises from "
            f"{percent(scaling[(scaling.dataset == 'aws') & (scaling.n == 9)].iloc[0].strict_edge_and_final_success)} "
            "to "
            f"{percent(scaling[(scaling.dataset == 'aws') & (scaling.n == 101)].iloc[0].strict_edge_and_final_success)} on AWS, and from "
            f"{percent(scaling[(scaling.dataset == 'wp') & (scaling.n == 9)].iloc[0].strict_edge_and_final_success)} "
            "to "
            f"{percent(scaling[(scaling.dataset == 'wp') & (scaling.n == 101)].iloc[0].strict_edge_and_final_success)} on WP."
        ),
        "",
        "## Multi-observer ablation",
        "",
    ])
    table = composition.copy()
    table["strict edge"] = table.apply(lambda r: ci(
        r.strict_edge, r.strict_edge_ci95_low, r.strict_edge_ci95_high), axis=1)
    table["final F<T"] = table.apply(lambda r: ci(
        r.final_F_before_T, r.final_F_before_T_ci95_low,
        r.final_F_before_T_ci95_high), axis=1)
    table["edge and final"] = table.apply(lambda r: ci(
        r.strict_edge_and_final, r.strict_edge_and_final_ci95_low,
        r.strict_edge_and_final_ci95_high), axis=1)
    lines.extend([
        markdown_table(table, ["dataset", "arm", "strict edge", "final F<T", "edge and final"]),
        "",
        (
            "With pair-swap reports, the operational independently trained "
            "singleton-to-joint strict-edge change is "
            f"{percent(composition[(composition.dataset == 'aws') & (composition.arm == 'independent_singleton_reports')].iloc[0].strict_edge)} to "
            f"{percent(composition[(composition.dataset == 'aws') & (composition.arm == 'five_observer_joint')].iloc[0].strict_edge)} on AWS and "
            f"{percent(composition[(composition.dataset == 'wp') & (composition.arm == 'independent_singleton_reports')].iloc[0].strict_edge)} to "
            f"{percent(composition[(composition.dataset == 'wp') & (composition.arm == 'five_observer_joint')].iloc[0].strict_edge)} on WP."
        ),
        "",
        "The `oracle_singleton_reports_diagnostic` arm chooses the best singleton after seeing the execution and is not an operational attacker claim. The two independently trained singleton arms are the matched operational baselines.",
        "",
        "### Strict singleton-union separation",
        "",
    ])
    table = separation.copy()
    table["estimate"] = table.apply(
        lambda r: ci(r.value, r.ci95_low, r.ci95_high), axis=1
    )
    lines.extend([
        markdown_table(table, ["dataset", "separation", "estimate"]),
        "",
        "## Placement sensitivity",
        "",
    ])
    table = placement.copy()
    table["strict edge"] = table.apply(lambda r: ci(
        r.strict_direct_edge_FT, r.strict_direct_edge_FT_ci95_low,
        r.strict_direct_edge_FT_ci95_high), axis=1)
    table["edge and final"] = table.apply(lambda r: ci(
        r.strict_edge_and_final_success,
        r.strict_edge_and_final_success_ci95_low,
        r.strict_edge_and_final_success_ci95_high), axis=1)
    lines.extend([
        markdown_table(table, ["dataset", "placement", "strict edge", "edge and final"]),
        "",
        "Placement is a sensitivity result: only corrupted-replica locations change within each matched deployment; honest replicas, workload, trace sample, reports, and public random tape remain paired.",
        "",
        "## Reaction-delay sensitivity",
        "",
    ])
    table = rho.copy()
    table["strict edge"] = table.apply(lambda r: ci(
        r.strict_direct_edge_FT, r.strict_direct_edge_FT_ci95_low,
        r.strict_direct_edge_FT_ci95_high), axis=1)
    table["edge and final"] = table.apply(lambda r: ci(
        r.strict_edge_and_final_success,
        r.strict_edge_and_final_success_ci95_low,
        r.strict_edge_and_final_success_ci95_high), axis=1)
    lines.extend([
        markdown_table(table, ["dataset", "rho_ms", "strict edge", "edge and final"]),
        "",
        "The delay sweep quantifies the precomputation boundary directly. It should be reported alongside, rather than replaced by, the rho=0 worst-case endpoint.",
        "",
        "## Guard",
        "",
        "The headline Guard workload now exactly replays the Section IV n=21 joint attack before applying admission control. AWS uses the Section IV global-random layout and WP uses its latency-clustered layout; both permit distinct corrupted and honest replica identities to share a measured regional profile. Placement is trained on history only. The victim, T/F arrivals, 300-transaction workload, reports, graph, and public Unspool tape are paired with Section IV trial by trial; `paper_guard_section4_match.csv` audits this equality.",
        "",
    ])
    table = guard.copy()
    for output in (
        "raw_strict_edge", "carry_given_edge", "raw_final_F_before_T",
        "guarded_all_trial_residual", "benign_pair_false_carry",
        "benign_batch_any_false_carry", "all_trial_carry_rate",
        "residual_given_carry",
    ):
        table[output] = table.apply(lambda r, name=output: ci(
            r[name], r[f"{name}_ci95_low"], r[f"{name}_ci95_high"]), axis=1)
    lines.extend([
        markdown_table(table, [
            "dataset", "bound_mode", "raw_strict_edge", "carry_given_edge",
            "all_trial_carry_rate", "raw_final_F_before_T",
            "guarded_all_trial_residual", "residual_given_carry",
            "benign_pair_false_carry", "benign_batch_any_false_carry",
        ]),
        "",
        "For WP, `previous-hour` freezes bounds from the immediately preceding hourly snapshot. AWS has no timestamps, so its corresponding row is a held-out even-sample/odd-sample partition and must not be described as temporal validation.",
        "",
        "Full component data, including profile-only, capacity-only, full guard, and carry-every-reverse-edge, are in `paper_guard_ablation.csv`.",
        "",
        "## Scope",
        "",
        "The attack rows use the legal worst-case reporter set and five pair-swap Byzantine reports for n=21 (scaled to f for n=9/101). This is the relevant BFT safety endpoint, not an average reporter-selection claim. Reaction delay is modeled transport-plane delay only; no SHA-256 or unrelated transaction construction is inserted.",
        "",
        "The same-hour guard row is an in-window theorem/conformance endpoint. The held-out row is the deployment experiment: bounds are frozen before the execution partition. Neither a failed edge nor a carried transaction is silently counted as an attack success; strict edge, final order, and their conjunction are reported separately.",
        "",
    ])
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def write_manuscript_notes(root):
    text = """# Manuscript Update Notes

The frozen artifact closes the following draft placeholders.

1. Populate every AWS/WP `n = 9, 21, 101` point in Figure 1 from
   `paper_scaling.csv`. Use all three metrics and their hierarchical-bootstrap
   intervals; remove the hollow WP `n=101` pilot marker.
2. Populate Table 2 from `paper_composition.csv`. The operational singleton is
   `independent_singleton_*`; keep `oracle_singleton_reports_diagnostic`
   explicitly ex post.
3. Populate Figure 2 from `paper_composition_separation.csv`, including the
   conjunction with final output.
4. Replace the draft statement that placement rows are separately sampled.
   `paper_placement.csv` uses matched deployment, workload, execution sample,
   reporter set, and random tape.
5. Replace "positive end-to-end response delay remains to be completed" with
   the `rho = 0, 1, 2, 5 ms` sensitivity in `paper_rho.csv`. Preserve `rho=0`
   as the attacker-favorable endpoint rather than a sole practicality claim.
6. Populate the Guard main table from `paper_guard.csv` and the component table
   from `paper_guard_ablation.csv`. Call WP `previous-hour` temporal held-out
   calibration; call AWS an even/odd sample-isolation result, not future-time
   validation. The formal Guard rows replay the Section IV n=21 global-joint
   attack exactly before applying admission control. The earlier no-colocation
   workload remains only as a constrained-placement sensitivity under `guard/`.
7. Keep the current distinction between all-trial residual success and
   residual conditional on carry. Carrying F to the next epoch fixes T before
   F by epoch order; residual all-trial successes arise only when the guard did
   not carry the transaction.

Do not describe `final F<T` as caused solely by the direct edge. The strict
edge, final output, and conjunction remain separate throughout the artifact.
"""
    (root / "MANUSCRIPT_UPDATE_NOTES.md").write_text(text, encoding="utf-8")


def write_guard_reevaluation(root, guard, legacy, match):
    lines = [
        "# Matched Guard Re-evaluation",
        "",
        "This document replaces the earlier Guard headline numbers. The old run "
        "excluded every location occupied by an honest replica from corrupted "
        "placement. That restriction is not required by the attack model and left "
        "only five AWS candidates. The corrected run permits co-located replica "
        "identities and replays the Section IV n=21 joint attack exactly.",
        "",
        "## What changed",
        "",
        "- AWS uses `global-random`; WP uses `latency-clustered`, matching Section IV.",
        "- `allow_corrupted_region_colocation=true`; all measured trace locations are placement candidates.",
        "- T/F execution samples, client assignments, Poisson releases, reporter order, pair-processing order, and public Unspool permutations use the Section IV keys.",
        "- Guard bounds and carryover are the only additional operations.",
        "- The old `guard/` tree is retained as a constrained-placement sensitivity; the corrected formal rows are under `guard-matched/`.",
        "",
        "## Trial-level match audit",
        "",
        markdown_table(match, [
            "dataset", "bound_mode", "paired_trials",
            "victim_exact_matches", "strict_edge_exact_matches",
            "direct_edge_exact_matches", "w_FT_exact_matches",
            "w_TF_exact_matches", "final_rate_exact_matches",
            "all_fields_exact_matches",
        ]),
        "",
        "Every equality above is evaluated on the same `(deployment, trial)` key. A formal run is valid only if all fields match in all 400 trials.",
        "",
        "## Replacement results",
        "",
    ]
    replacement = guard.copy()
    for metric in (
        "raw_strict_edge", "carry_given_edge", "all_trial_carry_rate",
        "raw_final_F_before_T", "guarded_all_trial_residual",
        "residual_given_carry", "benign_pair_false_carry",
        "benign_batch_any_false_carry",
    ):
        replacement[metric] = replacement.apply(
            lambda row, name=metric: ci(
                row[name], row[f"{name}_ci95_low"],
                row[f"{name}_ci95_high"],
            ), axis=1,
        )
    lines.extend([
        markdown_table(replacement, [
            "dataset", "bound_mode", "raw_strict_edge", "carry_given_edge",
            "all_trial_carry_rate", "raw_final_F_before_T",
            "guarded_all_trial_residual", "residual_given_carry",
            "benign_pair_false_carry", "benign_batch_any_false_carry",
        ]),
        "",
        "## Old-to-new comparison",
        "",
    ])
    comparison_rows = []
    for _, new_row in guard.iterrows():
        old_row = legacy[
            (legacy.dataset == new_row.dataset)
            & (legacy.bound_mode == new_row.bound_mode)
        ].iloc[0]
        comparison_rows.append({
            "dataset": new_row.dataset,
            "bound_mode": new_row.bound_mode,
            "old_raw_edge": percent(old_row.raw_strict_edge),
            "matched_raw_edge": percent(new_row.raw_strict_edge),
            "old_pair_false_carry": percent(
                old_row.benign_pair_false_carry
            ),
            "matched_pair_false_carry": percent(
                new_row.benign_pair_false_carry
            ),
            "old_batch_any_false_carry": percent(
                old_row.benign_batch_any_false_carry
            ),
            "matched_batch_any_false_carry": percent(
                new_row.benign_batch_any_false_carry
            ),
        })
    lines.extend([
        markdown_table(pd.DataFrame(comparison_rows), [
            "dataset", "bound_mode", "old_raw_edge", "matched_raw_edge",
            "old_pair_false_carry", "matched_pair_false_carry",
            "old_batch_any_false_carry", "matched_batch_any_false_carry",
        ]),
        "",
        "Only the Guard tables, Guard figure, component ablation, and Guard C++ conformance cases are replaced. Cross-scale, composition, placement, and reaction-delay results are unaffected by this correction.",
        "",
    ])
    (root / "GUARD_MATCHED_REEVALUATION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    root = args.result_root.resolve()
    scaling = collect_scaling(root)
    composition, separation = collect_composition(root)
    placement = collect_placement(root)
    rho = collect_rho(root)
    guard, components = collect_guard(root, "guard-matched")
    guard = align_guard_raw_intervals(guard, scaling)
    legacy_guard, _ = collect_guard(root, "guard")
    guard_match = collect_guard_section4_match(root)
    outputs = {
        "paper_scaling.csv": scaling,
        "paper_composition.csv": composition,
        "paper_composition_separation.csv": separation,
        "paper_placement.csv": placement,
        "paper_rho.csv": rho,
        "paper_guard.csv": guard,
        "paper_guard_ablation.csv": components,
        "paper_guard_section4_match.csv": guard_match,
    }
    for filename, frame in outputs.items():
        frame.to_csv(root / filename, index=False)
    write_report(
        root, scaling, composition, separation, placement, rho, guard, components
    )
    write_manuscript_notes(root)
    write_guard_reevaluation(root, guard, legacy_guard, guard_match)
    print(root / "RESULTS.md")


if __name__ == "__main__":
    main()
