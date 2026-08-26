#!/usr/bin/env python3

import importlib.util
import pathlib
import sys
import unittest

import numpy as np


MODULE_PATH = pathlib.Path(__file__).with_name(
    "evaluate_corruption_budget_ablation.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("corruption_budget", MODULE_PATH)
BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUDGET)


class CorruptionBudgetAblationTest(unittest.TestCase):
    def test_every_arm_uses_exact_actual_corruption_budget(self):
        for corruptions in range(1, 6):
            for _, parameter_fn in BUDGET.STRATEGIES:
                observers, reports = parameter_fn(corruptions)
                self.assertEqual(max(observers, reports), corruptions)

    def test_only_selected_corrupted_reporters_are_forged(self):
        deployment = {"reporter_order": list(range(16))}
        snapshot = {
            "corrupted": [0, 1, 2],
            "t_arrivals": np.zeros(16),
        }
        f_arrivals = np.ones(16)
        f_arrivals[3:11] = -1.0

        truthful = BUDGET.evaluate_pair(
            deployment, snapshot, f_arrivals, []
        )
        one_report = BUDGET.evaluate_pair(
            deployment, snapshot, f_arrivals, [0]
        )
        two_reports = BUDGET.evaluate_pair(
            deployment, snapshot, f_arrivals, [0, 1]
        )

        self.assertEqual((truthful["w_FT"], truthful["w_TF"]), (8, 8))
        self.assertEqual((one_report["w_FT"], one_report["w_TF"]), (9, 7))
        self.assertEqual((two_reports["w_FT"], two_reports["w_TF"]), (10, 6))
        self.assertEqual(truthful["strict_edge"], 0)
        self.assertEqual(one_report["strict_edge"], 1)
        self.assertEqual(two_reports["strict_edge"], 1)
        self.assertEqual(two_reports["selected_honest_support"], 8)


if __name__ == "__main__":
    unittest.main()
