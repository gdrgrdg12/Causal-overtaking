#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import re
import subprocess

import pandas as pd

import run_original_unspool_conformance as original


CASE_PATTERN = re.compile(
    r"(?P<dataset>aws|wp)-d(?P<deployment>\d+)-"
    r"t(?P<trial>\d+)-.*\.case$"
)


def replay(binary, case_dir, work_dir, row):
    case_file = row.unspool_conformance_case
    match = CASE_PATTERN.fullmatch(case_file)
    if not match:
        raise ValueError(f"invalid case name: {case_file}")
    case_path = os.path.join(case_dir, case_file)
    paths = original.materialize(case_path, work_dir)
    completed = subprocess.run(
        [
            binary, paths[0],
            "--fixed-quorum-file", paths[1],
            "--pair-order-file", paths[2],
            "--public-order-file", paths[3],
        ],
        check=False, text=True, capture_output=True,
    )
    total = original.TOTAL_PATTERN.search(completed.stdout)
    if completed.returncode != 0 or not total:
        raise RuntimeError(
            f"C++ replay failed for {case_file}: {completed.stderr[-1000:]}"
        )
    successes, repetitions = map(int, total.groups())
    observed = successes / repetitions
    expected = float(row.raw_final_F_before_T_rate)
    return {
        "deployment": int(match.group("deployment")),
        "dataset": match.group("dataset"),
        "trial": int(match.group("trial")),
        "case_file": case_file,
        "repetitions": repetitions,
        "python_final_F_before_T_rate": expected,
        "cpp_final_F_before_T_rate": observed,
        "absolute_difference": abs(expected - observed),
        "exact_match": int(abs(expected - observed) < 1e-12),
    }


def run(args):
    if not 1 <= args.workers <= 5:
        raise ValueError("workers must be between 1 and 5")
    trials = pd.read_csv(os.path.join(args.result_dir, "trial_summary.csv"))
    selected = trials[trials.unspool_conformance_case.fillna("") != ""]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:
        futures = [
            pool.submit(
                replay, args.binary,
                os.path.join(args.result_dir, "unspool-conformance"),
                args.work_dir, row,
            )
            for _, row in selected.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    output = pd.DataFrame(rows).sort_values(["dataset", "deployment", "trial"])
    output.to_csv(
        os.path.join(args.result_dir, "original_unspool_conformance.csv"),
        index=False,
    )
    if len(output) == 0 or not output.exact_match.all():
        raise SystemExit("instrumented artifact C++ Unspool conformance failed")
    print(
        "instrumented artifact C++ Unspool exact matches: "
        f"{len(output)}/{len(output)}"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--workers", type=int, default=5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
