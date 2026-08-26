#!/usr/bin/env python3
"""Build coherent hourly WonderProxy matrices from the official raw trace."""

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nearest_fill(values):
    """Fill missing hours on each path from the nearest measured hour."""
    hours, paths = values.shape
    filled = values.copy()
    for path_index in range(paths):
        column = filled[:, path_index]
        valid = np.flatnonzero(np.isfinite(column))
        if len(valid) == 0:
            raise ValueError(f"path {path_index} has no measurements")
        missing = np.flatnonzero(~np.isfinite(column))
        if len(missing) == 0:
            continue
        insertion = np.searchsorted(valid, missing)
        left_pos = np.maximum(insertion - 1, 0)
        right_pos = np.minimum(insertion, len(valid) - 1)
        left = valid[left_pos]
        right = valid[right_pos]
        use_right = np.abs(right - missing) < np.abs(missing - left)
        nearest = np.where(use_right, right, left)
        column[missing] = column[nearest]
    return filled


def run(args):
    simplified = pd.read_csv(args.simplified_file)
    locations = sorted({str(value) for value in simplified.source.unique()})
    location_set = {int(value) for value in locations}
    index = {location: position for position, location in enumerate(locations)}

    columns = [
        "source", "destination", "timestamp", "min", "avg", "max", "mdev"
    ]
    selected = []
    for chunk in pd.read_csv(args.raw_file, usecols=columns, chunksize=250_000):
        keep = (
            chunk.source.isin(location_set)
            & chunk.destination.isin(location_set)
        )
        selected.append(chunk.loc[keep])
    frame = pd.concat(selected, ignore_index=True)

    if len(frame) != len(simplified):
        raise ValueError(
            "raw trace projection and simplified trace have different row counts"
        )
    if not np.array_equal(
        frame.source.to_numpy(dtype=int), simplified.source.to_numpy(dtype=int)
    ):
        raise ValueError("source order differs from the simplified artifact")
    if not np.array_equal(
        frame.destination.to_numpy(dtype=int),
        simplified.destination.to_numpy(dtype=int),
    ):
        raise ValueError("destination order differs from the simplified artifact")
    if not np.allclose(
        frame.avg.to_numpy(dtype=float),
        simplified.avg.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("RTT values differ from the simplified artifact")

    frame["hour"] = pd.to_datetime(frame.timestamp, utc=True).dt.floor("h")
    hours = pd.DatetimeIndex(sorted(frame.hour.unique()))
    hour_index = {hour: position for position, hour in enumerate(hours)}
    shape = (len(hours), len(locations), len(locations))
    avg_owd = np.full(shape, np.nan, dtype=np.float32)
    min_owd = np.full(shape, np.nan, dtype=np.float32)
    max_owd = np.full(shape, np.nan, dtype=np.float32)
    mdev_owd = np.full(shape, np.nan, dtype=np.float32)
    failed_measurements = 0

    for row in frame.itertuples(index=False):
        if min(row.min, row.avg, row.max, row.mdev) < 0.0:
            failed_measurements += 1
            continue
        hour = pd.Timestamp(row.hour)
        source = index[str(row.source)]
        destination = index[str(row.destination)]
        min_owd[hour_index[hour], source, destination] = float(row.min) * 0.5
        avg_owd[hour_index[hour], source, destination] = float(row.avg) * 0.5
        max_owd[hour_index[hour], source, destination] = float(row.max) * 0.5
        mdev_owd[hour_index[hour], source, destination] = float(row.mdev) * 0.5

    missing = ~np.isfinite(avg_owd)
    for label, values in [
        ("min", min_owd), ("max", max_owd), ("mdev", mdev_owd)
    ]:
        if not np.array_equal(missing, ~np.isfinite(values)):
            raise ValueError(f"avg and {label} missing-path masks differ")
    min_owd = nearest_fill(min_owd.reshape(len(hours), -1)).reshape(shape)
    avg_owd = nearest_fill(avg_owd.reshape(len(hours), -1)).reshape(shape)
    max_owd = nearest_fill(max_owd.reshape(len(hours), -1)).reshape(shape)
    mdev_owd = nearest_fill(mdev_owd.reshape(len(hours), -1)).reshape(shape)

    metadata = {
        "source": (
            "WonderProxy pings-2020-07-19-2020-07-20.csv.gz; hourly "
            "ping -c 30 summaries"
        ),
        "raw_sha256": sha256(args.raw_file),
        "simplified_sha256": sha256(args.simplified_file),
        "locations": len(locations),
        "hours": len(hours),
        "path_slots": int(np.prod(shape)),
        "measured_path_slots": int(np.sum(~missing)),
        "imputed_path_slots": int(np.sum(missing)),
        "failed_measurements": failed_measurements,
        "absent_measurements": int(np.sum(missing)) - failed_measurements,
        "imputation": "nearest available hourly measurement on the same path",
        "delay_model": (
            "current path center avg RTT / 2; short-window envelope "
            "[min RTT / 2, max RTT / 2]; mdev RTT / 2 sampling scale"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    np.savez_compressed(
        args.output_file,
        locations=np.asarray(locations, dtype="U16"),
        hours=np.asarray([str(hour) for hour in hours], dtype="U32"),
        min_owd=min_owd,
        avg_owd=avg_owd,
        max_owd=max_owd,
        mdev_owd=mdev_owd,
        imputed=missing,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2))
    print(f"created: {args.output_file}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True)
    parser.add_argument(
        "--simplified_file", default="./data-ping/wp-simplified.csv"
    )
    parser.add_argument(
        "--output_file", default="./data-ping/wp-hourly-2020-07-19-20.npz"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
