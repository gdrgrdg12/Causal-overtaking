#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np


MODULE_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
SPEC = importlib.util.spec_from_file_location(
    "corruption_guard_background",
    MODULE_DIR / "evaluate_corruption_guard_background.py",
)
BACKGROUND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKGROUND)


class FakeWindow:
    def sample(self, epoch, source, replica, *seed_parts):
        return float(int(source[1:]) + int(replica[1:]))


class FakeMatchedTrace:
    dataset = "wp"

    def sample(self, phase, source, replica, seed, *parts):
        del phase, seed, parts
        return float(int(source[1:]) + int(replica[1:]))


class CorruptionGuardBackgroundTest(unittest.TestCase):
    def test_section4_background_reuses_generator_and_sample_keys(self):
        trace = FakeMatchedTrace()
        deployment = {
            "dataset": "wp",
            "deployment": 0,
            "clients": ["c0", "c1", "c2"],
            "replicas": ["r0", "r1"],
        }
        args = SimpleNamespace(seed=19, background_txs=8)
        direct = BACKGROUND.matched.generate_background(
            trace, deployment, 0, args
        )
        _, _, arrivals = BACKGROUND.generate_section4_background(
            trace, deployment, 0, 0, 8, 19
        )
        for replica_index, replica in enumerate(deployment["replicas"]):
            expected = dict(direct[replica])
            for tx_index in range(8):
                self.assertEqual(
                    arrivals[tx_index, replica_index], expected[f"B{tx_index:03d}"]
                )

    def test_trial_dataset_is_not_hard_coded(self):
        source = (MODULE_DIR / "evaluate_corruption_guard_background.py").read_text()
        self.assertNotIn('"dataset": "wp"', source)

    def test_capacity_context_uses_frozen_bound_epoch(self):
        source = (MODULE_DIR / "evaluate_corruption_guard_background.py").read_text()
        self.assertIn(
            'window, snapshot["bound_epoch"], args.uncertainty_scale', source
        )

    def test_background_matches_per_client_poisson_processes(self):
        deployment = {
            "clients": [f"c{index}" for index in range(15)],
            "replicas": ["r0", "r1"],
        }
        snapshot = {"epoch": 0}
        releases, sources, arrivals = BACKGROUND.generate_background(
            FakeWindow(), snapshot, deployment, 0, 0, 300, 17
        )

        self.assertEqual(len(releases), 300)
        self.assertEqual(arrivals.shape, (300, 2))
        for client in deployment["clients"]:
            indices = [i for i, source in enumerate(sources) if source == client]
            self.assertEqual(len(indices), 20)
            self.assertTrue(np.all(np.diff(releases[indices]) > 0))

    def test_uneven_count_is_distributed_like_original_generator(self):
        deployment = {
            "clients": [f"c{index}" for index in range(3)],
            "replicas": ["r0"],
        }
        _, sources, _ = BACKGROUND.generate_background(
            FakeWindow(), {"epoch": 0}, deployment, 0, 0, 8, 19
        )
        counts = [sources.count(client) for client in deployment["clients"]]
        self.assertEqual(counts, [3, 3, 2])

    def test_non_deferred_transaction_keeps_raw_epoch_outcomes(self):
        raw = [{"final_ft": 1, "same_scc": 1, "ft_scc_size": 3}]
        result = BACKGROUND.apply_cross_epoch_admission(raw, deferred=0)
        self.assertEqual(result, raw)
        self.assertIsNot(result[0], raw[0])

    def test_deferred_transaction_is_absent_until_next_epoch(self):
        raw = [
            {"final_ft": seed % 2, "same_scc": 1, "ft_scc_size": 3}
            for seed in range(50)
        ]
        result = BACKGROUND.apply_cross_epoch_admission(raw, deferred=1)
        self.assertTrue(all(outcome["final_ft"] == 0 for outcome in result))
        self.assertTrue(all(outcome["same_scc"] == 0 for outcome in result))
        self.assertTrue(all(outcome["ft_scc_size"] == 0 for outcome in result))


if __name__ == "__main__":
    unittest.main()
