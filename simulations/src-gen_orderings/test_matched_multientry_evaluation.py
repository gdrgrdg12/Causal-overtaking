#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import types
import unittest
from collections import Counter


MODULE_PATH = pathlib.Path(__file__).with_name("matched_multientry_evaluation.py")
SPEC = importlib.util.spec_from_file_location("matched_multientry", MODULE_PATH)
MATCHED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATCHED)


def report(*transactions):
    return [(tx, float(index)) for index, tx in enumerate(transactions)]


class WeightSemanticsTest(unittest.TestCase):
    def test_left_only_counts_as_precedence(self):
        orderings = [report("F"), report("T"), report("F", "T"), report("T", "F")]
        self.assertEqual(MATCHED.pair_weights(orderings, range(4)), (2, 2, 2))

    def test_neither_present_counts_for_neither_direction(self):
        orderings = [report("B"), report("F"), report("T")]
        self.assertEqual(MATCHED.pair_weights(orderings, range(3)), (1, 1, 0))

    def test_tie_follows_first_report_pair_order(self):
        orderings = [
            report("F", "T", "B"),
            report("T", "F", "B"),
            report("F", "T", "B"),
            report("T", "F", "B"),
        ]
        tx_list, weights, adjacency = MATCHED.build_graph(orderings, range(4), 2)
        f_index = tx_list.index("F")
        t_index = tx_list.index("T")
        self.assertEqual(int(weights[f_index, t_index]), 2)
        self.assertEqual(int(weights[t_index, f_index]), 2)
        self.assertTrue(adjacency[f_index, t_index])
        self.assertFalse(adjacency[t_index, f_index])

        tx_list, _, adjacency = MATCHED.build_graph(
            orderings, range(4), 2, ["T", "F", "B"]
        )
        f_index = tx_list.index("F")
        t_index = tx_list.index("T")
        self.assertFalse(adjacency[f_index, t_index])
        self.assertTrue(adjacency[t_index, f_index])

    def test_presence_classification_for_n21_f5(self):
        self.assertEqual(MATCHED.presence_state(5, 21), "blank")
        self.assertEqual(MATCHED.presence_state(6, 21), "shaded")
        self.assertEqual(MATCHED.presence_state(10, 21), "shaded")
        self.assertEqual(MATCHED.presence_state(11, 21), "solid")

    def test_byzantine_pair_swap_preserves_background_slots(self):
        original = report("T", "B0", "B1", "F", "B2")
        forged = MATCHED.forge_byzantine_report(original, "pair-swap")

        self.assertEqual([tx for tx, _ in forged], ["F", "B0", "B1", "T", "B2"])
        self.assertEqual(
            [time for _, time in forged], [time for _, time in original]
        )
        self.assertEqual(
            [(index, item) for index, item in enumerate(forged) if item[0].startswith("B")],
            [(index, item) for index, item in enumerate(original) if item[0].startswith("B")],
        )

    def test_byzantine_pair_swap_leaves_existing_f_before_t_unchanged(self):
        original = report("B0", "F", "B1", "T")
        self.assertEqual(
            MATCHED.forge_byzantine_report(original, "pair-swap"), original
        )

    def test_byzantine_prefix_reproduces_legacy_model(self):
        original = report("T", "B0", "F", "B1")
        self.assertEqual(
            [tx for tx, _ in MATCHED.forge_byzantine_report(original, "prefix")],
            ["F", "T", "B0", "B1"],
        )


class PairingTest(unittest.TestCase):
    def test_logical_replica_alias_uses_physical_latency_location(self):
        logical = MATCHED.logical_replica_location("Ohio", 17)
        self.assertNotEqual(logical, "Ohio")
        self.assertEqual(MATCHED.physical_location(logical), "Ohio")

    def test_colocated_replicas_keep_distinct_identities(self):
        replicas = MATCHED.ensure_unique_replica_identities(
            ["Ohio", "Paris", "Ohio"]
        )
        self.assertEqual(len(set(replicas)), 3)
        self.assertEqual(
            [MATCHED.physical_location(value) for value in replicas],
            ["Ohio", "Paris", "Ohio"],
        )

    def test_plain_dissemination_preserves_t_before_f_on_exact_tie(self):
        deployment = {"replicas": ["r0"]}
        args = types.SimpleNamespace()
        result = MATCHED.apply_dissemination(
            None, deployment, 0, {"r0": [("T", 5.0)]},
            {"r0": 5.0}, "plain", args,
        )
        self.assertEqual([tx for tx, _ in result["r0"]], ["T", "F"])

    def test_hourly_snapshot_uses_one_matrix_and_splits_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.npz"
            matrices = MATCHED.np.asarray([
                [[0.0, 1.0], [2.0, 0.0]],
                [[0.0, 10.0], [20.0, 0.0]],
                [[0.0, 3.0], [4.0, 0.0]],
                [[0.0, 30.0], [40.0, 0.0]],
            ])
            MATCHED.np.savez_compressed(
                path,
                locations=MATCHED.np.asarray(["a", "b"]),
                hours=MATCHED.np.asarray(["h0", "h1", "h2", "h3"]),
                avg_owd=matrices,
                metadata=MATCHED.np.asarray(json.dumps({"kind": "test"})),
            )
            trace = MATCHED.LatencyTrace(str(path), "half", 0.15)
            trace.set_snapshot_context("history", 7, "train", 0)
            trace.set_snapshot_context("execution", 7, "trial", 0)

            history = trace.sample("history", "a", "b", 7, "x")
            execution = trace.sample("execution", "a", "b", 7, "x")
            self.assertIn(history, {1.0, 3.0})
            self.assertIn(execution, {10.0, 30.0})
            self.assertEqual(
                history, trace.sample("history", "a", "b", 99, "other")
            )
            self.assertEqual(trace.historical_distance("a", "b"), 2.5)

    def test_stable_rng_is_keyed_not_call_ordered(self):
        first = MATCHED.stable_rng(7, "trial", 3, "F").random()
        MATCHED.stable_rng(7, "unrelated").random()
        second = MATCHED.stable_rng(7, "trial", 3, "F").random()
        self.assertEqual(first, second)

    def test_union_is_coordinatewise_minimum(self):
        arrivals = {
            "a": {"r0": 3.0, "r1": 8.0},
            "b": {"r0": 5.0, "r1": 2.0},
        }
        self.assertEqual(
            MATCHED.union_arrivals(arrivals, ["a", "b"]),
            {"r0": 3.0, "r1": 2.0},
        )

    def test_forward_edge_without_reverse_path_is_finally_deterministic(self):
        transactions = ["F", "T", "B"]
        adjacency = MATCHED.np.zeros((3, 3), dtype=MATCHED.np.bool_)
        adjacency[0, 1] = True
        self.assertFalse(MATCHED.reverse_path(adjacency, 1, 0))
        for public_seed in range(100):
            outcome = MATCHED.finalize_graph(
                transactions, adjacency, public_seed
            )
            self.assertEqual(outcome["final_ft"], 1)

    def test_explicit_public_permutation_matches_seeded_permutation(self):
        transactions = ["F", "T", "B"]
        adjacency = MATCHED.np.zeros((3, 3), dtype=MATCHED.np.bool_)
        adjacency[0, 1] = True
        seed = 91
        permutation = MATCHED.public_permutation(len(transactions), seed)
        seeded = MATCHED.finalize_graph(transactions, adjacency, seed)
        explicit = MATCHED.finalize_graph(
            transactions, adjacency, permutation=permutation
        )
        self.assertEqual(seeded, explicit)

    def test_pair_correlated_asymmetry_ignores_transmission_purpose(self):
        trace = MATCHED.LatencyTrace.__new__(MATCHED.LatencyTrace)
        trace.one_way_mode = "bounded_pair"
        trace.asymmetry_bound = 0.15
        first = trace.one_way_factor(
            7, ("execution", "a", "b", "observe", 1)
        )
        second = trace.one_way_factor(
            7, ("execution", "a", "b", "submit", 99)
        )
        self.assertEqual(first, second)

    def test_corrupted_relay_reuses_replica_t_arrival(self):
        class Trace:
            def sample(self, phase, source, destination, seed, *key):
                return {("c0", "entry"): 8.0, ("c1", "entry"): 1.0}[
                    (source, destination)
                ]

        deployment = {
            "replicas": ["c0", "c1", "r2"],
            "reporter_order": [0, 1, 2],
        }
        args = types.SimpleNamespace(
            observation_model="corrupted-replica-relay",
            corrupted_observers=2,
            seed=7,
        )
        observed = MATCHED.observe_entry(
            Trace(), deployment, "victim", {"c0": 10.0, "c1": 20.0},
            {"id": "E000", "location": "entry"}, "execution", args,
            "trial", 0, 0,
        )
        self.assertEqual(observed, 18.0)

    def test_clustered_layout_corrupts_one_replica_per_region(self):
        class Trace:
            dataset = "wp"
            locations = [str(index) for index in range(30)]

            @staticmethod
            def historical_distance(source, destination):
                return abs(int(source) - int(destination))

        args = types.SimpleNamespace(
            seed=19,
            num_nodes=21,
            num_clients=5,
            entry_counts=[5],
            replica_layout="latency-clustered",
            replica_regions=5,
            observation_model="corrupted-replica-direct",
            corrupted_observers=5,
        )
        deployment = MATCHED.make_deployment(Trace(), 0, args)
        corrupted = MATCHED.corrupted_replica_indices(deployment, args)

        self.assertEqual(len(deployment["replicas"]), 21)
        self.assertEqual(len(set(deployment["replicas"])), 21)
        self.assertEqual(
            sorted(Counter(deployment["replica_regions"]).values()),
            [4, 4, 4, 4, 5],
        )
        self.assertEqual(
            {deployment["replica_regions"][index] for index in corrupted},
            set(range(5)),
        )
        self.assertEqual(deployment["reporter_order"][:5], corrupted)
        self.assertEqual(len(deployment["entries"]), 5)
        self.assertTrue(all(
            entry["source_kind"] == "corrupted-replica-direct"
            for entry in deployment["entries"]
        ))

    def test_hybrid_contains_relay_entries_and_direct_sources(self):
        class Trace:
            dataset = "wp"
            locations = [str(index) for index in range(30)]

            @staticmethod
            def historical_distance(source, destination):
                return abs(int(source) - int(destination))

        args = types.SimpleNamespace(
            seed=19,
            num_nodes=21,
            num_clients=5,
            entry_counts=[8],
            relay_entry_count=8,
            replica_layout="latency-clustered",
            replica_regions=5,
            observation_model="corrupted-replica-hybrid",
            corrupted_observers=5,
        )
        deployment = MATCHED.make_deployment(Trace(), 0, args)
        kinds = Counter(entry["source_kind"] for entry in deployment["entries"])
        self.assertEqual(kinds["relay-entry"], 8)
        self.assertEqual(kinds["corrupted-replica-direct"], 5)

    def test_corrupted_replica_direct_has_no_entry_relay(self):
        class Trace:
            def sample(self, *args, **kwargs):
                raise AssertionError("direct observation must not sample a relay")

        deployment = {
            "replicas": ["c0", "r1"],
            "reporter_order": [0, 1],
            "corrupted_replica_indices": [0],
        }
        args = types.SimpleNamespace(
            observation_model="corrupted-replica-direct",
            corrupted_observers=1,
            seed=7,
        )
        observed = MATCHED.observe_entry(
            Trace(), deployment, "victim", {"c0": 12.0, "r1": 20.0},
            {"id": "C000", "location": "c0", "source_replica_index": 0},
            "execution", args, "trial", 0,
        )
        self.assertEqual(observed, 12.0)

    def test_nonforging_corrupted_observers_are_not_honest_capacity(self):
        deployment = {
            "reporter_order": [0, 1, 2, 3],
            "corrupted_replica_indices": [0, 1],
        }
        args = types.SimpleNamespace(
            observation_model="corrupted-replica-direct",
            corrupted_observers=2,
            num_nodes=4,
        )
        self.assertEqual(
            MATCHED.honest_replica_indices(
                deployment, args, report_byzantine=set()
            ),
            [2, 3],
        )

    def test_strict_honest_support_accounts_for_report_forgery(self):
        self.assertEqual(
            MATCHED.strict_honest_support_required(21, 16, 5), 4
        )
        self.assertEqual(
            MATCHED.strict_honest_support_required(21, 16, 0), 9
        )
        self.assertEqual(
            MATCHED.strict_honest_support_required(101, 76, 25), 14
        )


if __name__ == "__main__":
    unittest.main()
