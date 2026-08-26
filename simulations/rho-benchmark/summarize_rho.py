#!/usr/bin/env python3

import argparse
import csv
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime, timezone


METRICS = (
    "parse_us",
    "build_hash_us",
    "sign_us",
    "serialize_us",
    "send_us",
    "rho_send_call_us",
    "rho_send_done_us",
    "rho_sink_us",
)


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    by_mode = {}
    for path in args.inputs:
        with open(path, newline="") as source:
            for row in csv.DictReader(source):
                mode = row["mode"]
                values = by_mode.setdefault(
                    mode, {metric: [] for metric in METRICS}
                )
                for metric in METRICS:
                    values[metric].append(float(row[metric]))

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as output:
        fieldnames = (
            "mode",
            "metric",
            "samples",
            "mean_us",
            "p50_us",
            "p90_us",
            "p95_us",
            "p99_us",
            "p999_us",
            "max_us",
        )
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for mode in sorted(by_mode):
            for metric in METRICS:
                values = by_mode[mode][metric]
                writer.writerow(
                    {
                        "mode": mode,
                        "metric": metric,
                        "samples": len(values),
                        "mean_us": f"{statistics.fmean(values):.3f}",
                        "p50_us": f"{percentile(values, 0.50):.3f}",
                        "p90_us": f"{percentile(values, 0.90):.3f}",
                        "p95_us": f"{percentile(values, 0.95):.3f}",
                        "p99_us": f"{percentile(values, 0.99):.3f}",
                        "p999_us": f"{percentile(values, 0.999):.3f}",
                        "max_us": f"{max(values):.3f}",
                    }
                )

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    try:
        lscpu_output = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        lscpu = {}
        for line in lscpu_output.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                lscpu[key.strip()] = value.strip()
    except (OSError, subprocess.SubprocessError):
        lscpu = {}
    with open(metadata_path, "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("key", "value"))
        writer.writerow(("generated_utc", datetime.now(timezone.utc).isoformat()))
        writer.writerow(("hostname", platform.node()))
        writer.writerow(("platform", platform.platform()))
        writer.writerow(("architecture", platform.machine()))
        writer.writerow(("cpu_vendor", lscpu.get("Vendor ID", "unknown")))
        writer.writerow(("cpu_model", lscpu.get("Model name", lscpu.get("Model", "unknown"))))
        writer.writerow(("cpu_count", os.cpu_count()))
        writer.writerow(("clock", "CLOCK_MONOTONIC_RAW"))
        writer.writerow(("transport", "persistent loopback TCP with TCP_NODELAY"))

    print(summary_path)


if __name__ == "__main__":
    main()
