#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name("evaluate_corruption_aware_guard.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("corruption_guard", MODULE_PATH)
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class CorruptionAwareGuardTest(unittest.TestCase):
    def test_deployment_args_preserve_matched_layout_and_colocation(self):
        args = SimpleNamespace(
            seed=7, rho_ms=0.0, replica_layout="global-random",
            replica_regions=5, allow_corrupted_region_colocation=True,
        )
        deployment = GUARD.deployment_args(args)
        self.assertEqual(deployment.replica_layout, "global-random")
        self.assertTrue(deployment.allow_corrupted_region_colocation)
        self.assertEqual(deployment.byzantine_reports, 5)

    def test_csv_trace_freezes_even_samples_and_executes_on_odd_samples(self):
        rows = []
        for source in ("a", "b"):
            for destination in ("a", "b"):
                values = [0.0, 0.0, 0.0, 0.0]
                if source != destination:
                    values = [10.0, 30.0, 14.0, 34.0]
                rows.extend({
                    "source": source,
                    "destination": destination,
                    "time": value,
                } for value in values)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "aws.csv"
            GUARD.pd.DataFrame(rows).to_csv(path, index=False)
            window = GUARD.WindowTrace(str(path))
            self.assertEqual(window.minimum[0, 0, 1], 5.0)
            self.assertEqual(window.maximum[0, 0, 1], 7.0)
            self.assertEqual(window.minimum[1, 0, 1], 15.0)
            self.assertEqual(window.maximum[1, 0, 1], 17.0)
            execution = window.sample(1, "a", "b", 7, "trial")
            self.assertIn(execution, {15.0, 17.0})

    def test_snapshot_sample_is_deterministic_and_bounded(self):
        window = GUARD.WindowTrace.__new__(GUARD.WindowTrace)
        window.index = {"a": 0, "b": 1}
        window.average = np.asarray([[[0.0, 10.0], [10.0, 0.0]]])
        window.mdev = np.asarray([[[0.0, 3.0], [3.0, 0.0]]])
        window.minimum = np.asarray([[[0.0, 8.0], [8.0, 0.0]]])
        window.maximum = np.asarray([[[0.0, 12.0], [12.0, 0.0]]])
        first = window.sample(0, "a", "b", 7, "trial")
        second = window.sample(0, "a", "b", 7, "trial")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 8.0)
        self.assertLessEqual(first, 12.0)

    def test_pair_swap_only_changes_reversed_byzantine_pair(self):
        t = np.asarray([1.0, 3.0, 5.0])
        f = np.asarray([2.0, 2.0, 6.0])
        byzantine = np.asarray([True, True, False])
        report_t, report_f = GUARD.apply_report_behavior(
            t, f, byzantine, "pair-swap"
        )
        np.testing.assert_array_equal(report_t, [2.0, 3.0, 5.0])
        np.testing.assert_array_equal(report_f, [1.0, 2.0, 6.0])

    def test_arrival_tie_matches_stable_t_before_f_order(self):
        t = np.asarray([5.0, 5.0])
        f = GUARD.encode_t_before_f_ties(t, [5.0, 4.0])
        self.assertGreater(f[0], t[0] + GUARD.EPS_MS)
        self.assertEqual(f[1], 4.0)
        self.assertEqual(GUARD.strict_edge(t, f)[:2], (1, 1))
        report_t, report_f = GUARD.apply_report_behavior(
            t, f, np.asarray([True, False]), "pair-swap"
        )
        self.assertLess(report_f[0], report_t[0] - GUARD.EPS_MS)

    def test_interval_overlap_does_not_mix_sources(self):
        lower = np.asarray([[0.0, 2.0], [5.0, 5.0]])
        upper = np.asarray([[0.0, 2.0], [5.0, 5.0]])
        support, source, delta = GUARD.best_interval_overlap(lower, upper)
        self.assertEqual(support, 2)
        self.assertEqual(source, 1)
        self.assertEqual(delta, 5.0)

    def test_strict_edge_uses_weight_majority(self):
        t = np.zeros(16)
        f = np.concatenate((np.full(9, -1.0), np.full(7, 1.0)))
        self.assertEqual(GUARD.strict_edge(t, f), (9, 7, 1))
        f = np.concatenate((np.full(8, -1.0), np.full(8, 1.0)))
        self.assertEqual(GUARD.strict_edge(t, f), (8, 8, 0))

    def test_trace_feasibility_uses_one_corruption_set(self):
        earliest = np.asarray([
            [-2.0, -2.0, -2.0],
            [5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0],
        ])
        subsets, _, _, feasible = GUARD.trace_feasible_subsets(
            earliest, reporters=[0, 1, 2], observed_gap=[-1.0, -1.0, -1.0],
            subset_size=1,
        )
        witnesses = [tuple(row) for row in subsets[feasible]]
        self.assertEqual(witnesses, [(0,)])

    def test_corrupted_report_is_not_subject_to_physical_floor(self):
        earliest = np.asarray([
            [5.0, -2.0],
            [5.0, 5.0],
        ])
        subsets, _, _, feasible = GUARD.trace_feasible_subsets(
            earliest, reporters=[0, 1], observed_gap=[-1.0, -1.0],
            subset_size=1,
        )
        witnesses = [tuple(row) for row in subsets[feasible]]
        self.assertEqual(witnesses, [(0,)])

    def test_previous_hour_bound_is_frozen_before_execution(self):
        self.assertEqual(GUARD.bound_epoch_for_execution(7, "same-hour"), 7)
        self.assertEqual(GUARD.bound_epoch_for_execution(7, "previous-hour"), 6)

    def test_actual_corruption_coverage_uses_one_fixed_set(self):
        context = {
            "subsets": np.asarray([[0, 1], [0, 2], [1, 2]]),
            "subset_earliest": np.asarray([
                [-2.0, -2.0, -2.0],
                [-2.0, 3.0, -2.0],
                [3.0, -2.0, -2.0],
            ]),
            "corrupt_reporters": np.asarray([
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ]),
        }
        self.assertEqual(
            GUARD.actual_corruption_set_covered(
                context, [-1.0, -1.0, -1.0], [0, 1]
            ),
            1,
        )
        self.assertEqual(
            GUARD.actual_corruption_set_covered(
                context, [-1.0, -1.0, -3.0], [0, 1]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
