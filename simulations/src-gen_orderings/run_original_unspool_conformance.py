#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import re
import subprocess

import pandas as pd


CASE_PATTERN = re.compile(
    r"(?P<dataset>[^-]+)-d(?P<deployment>\d+)-t(?P<trial>\d+)-"
    r"(?P<dissemination>plain|delayed)-(?P<arm>.+)\.case$"
)
TOTAL_PATTERN = re.compile(r"total:\s*(\d+)\s*/\s*(\d+)")


def split_values(line, prefix=""):
    if prefix and not line.startswith(prefix):
        raise ValueError(f"expected {prefix}, got {line[:40]}")
    return [value for value in line[len(prefix):].strip().split(",") if value]


def materialize(case_path, work_dir):
    with open(case_path) as file:
        lines = [line.rstrip("\n") for line in file]
    node_count = int(lines[0])
    reporters = split_values(lines[1])
    pair_order = split_values(lines[2], "PAIR:")
    if not lines[3].startswith("PUBLIC_COUNT:"):
        raise ValueError(f"case has no explicit public tape: {case_path}")
    public_count = int(lines[3].split(":", 1)[1])
    permutations = [
        [int(value) for value in split_values(lines[4 + index], "PERM:")]
        for index in range(public_count)
    ]
    reports = lines[4 + public_count:4 + public_count + node_count]
    if len(reports) != node_count:
        raise ValueError(f"case has {len(reports)} reports, expected {node_count}")

    stem = os.path.basename(case_path)[:-5]
    case_dir = os.path.join(work_dir, stem)
    os.makedirs(case_dir, exist_ok=True)
    input_path = os.path.join(case_dir, "input.csv")
    quorum_path = os.path.join(case_dir, "quorum.csv")
    pair_path = os.path.join(case_dir, "pair.csv")
    public_path = os.path.join(case_dir, "public.csv")
    with open(input_path, "w") as file:
        file.write(f"{node_count}\n")
        file.write("\n".join(reports) + "\n")
    with open(quorum_path, "w") as file:
        file.write(",".join(reporters) + "\n")
    with open(pair_path, "w") as file:
        file.write(",".join(pair_order) + "\n")
    with open(public_path, "w") as file:
        for permutation in permutations:
            file.write(",".join(pair_order[index] for index in permutation) + "\n")
    return input_path, quorum_path, pair_path, public_path


def replay_case(binary, case_path, work_dir, expected):
    paths = materialize(case_path, work_dir)
    command = [
        binary, paths[0],
        "--fixed-quorum-file", paths[1],
        "--pair-order-file", paths[2],
        "--public-order-file", paths[3],
    ]
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True
    )
    match = TOTAL_PATTERN.search(completed.stdout)
    if completed.returncode != 0 or not match:
        raise RuntimeError(
            f"replay failed for {case_path}: return={completed.returncode}\n"
            f"stdout={completed.stdout[-1000:]}\nstderr={completed.stderr[-1000:]}"
        )
    successes, repetitions = map(int, match.groups())
    observed = successes / repetitions
    keys = CASE_PATTERN.fullmatch(os.path.basename(case_path)).groupdict()
    return {
        **keys,
        "deployment": int(keys["deployment"]),
        "trial": int(keys["trial"]),
        "python_final_rate": expected,
        "original_unspool_final_rate": observed,
        "repetitions": repetitions,
        "absolute_difference": abs(expected - observed),
        "match_final_output": int(abs(expected - observed) < 1e-12),
    }


def run(args):
    if args.workers < 1 or args.workers > 5:
        raise ValueError("workers must be between 1 and 5")
    trials = pd.read_csv(os.path.join(args.result_dir, "trial_results.csv"))
    cases = []
    for filename in sorted(os.listdir(args.case_dir)):
        match = CASE_PATTERN.fullmatch(filename)
        if not match:
            continue
        keys = match.groupdict()
        selected = trials[
            (trials.dataset == keys["dataset"])
            & (trials.deployment == int(keys["deployment"]))
            & (trials.trial == int(keys["trial"]))
            & (trials.dissemination == keys["dissemination"])
            & (trials.arm == keys["arm"])
            & (trials.reporter_policy == "implementation")
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one trial row for {filename}")
        cases.append((
            os.path.join(args.case_dir, filename),
            float(selected.iloc[0].final_F_before_T_rate),
        ))

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                replay_case, args.binary, case_path, args.work_dir, expected
            )
            for case_path, expected in cases
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
    output = pd.DataFrame(rows).sort_values(
        ["dataset", "deployment", "trial", "dissemination", "arm"]
    )
    output.to_csv(args.output, index=False)
    if not output.match_final_output.all():
        raise SystemExit("original unspool final output mismatch")
    print(f"original unspool exact matches: {len(output)}/{len(output)}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--case_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=5)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
