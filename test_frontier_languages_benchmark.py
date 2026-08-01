from __future__ import annotations

import ast
import unittest

from benchmark_compact_languages import normalize_ast
from benchmark_frontier_languages import (
    EXPECTED_PAIRS,
    aggregate,
    cross_screen_market,
    frontier_pairs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class FrontierLanguagesBenchmarkTests(unittest.TestCase):
    def test_registry_is_fixed_complete_and_valid(self) -> None:
        pairs = frontier_pairs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(
            {pair.category for pair in pairs},
            {"array", "recurrence", "reduction", "scalar", "text"},
        )
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.gnu_apl.strip())
                self.assertTrue(pair.cjam.strip())
                self.assertTrue(pair.kona.strip())
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in frontier_pairs():
            with self.subTest(task_id=pair.task_id):
                expected = ast.unparse(compact_tree(ast.parse(pair.python)))
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_frontier_optimizations_are_emitted(self) -> None:
        encoded = {
            pair.task_id: transpile(pair.python, compact=True)
            for pair in frontier_pairs()
        }
        self.assertEqual(encoded["array/sort"], "$^#915372864")
        self.assertEqual(encoded["array/distinct"], "$?#31232415")
        self.assertEqual(encoded["array/dot_product"], "::@#123:#456")
        self.assertEqual(
            encoded["array/rotate_left"],
            "$values=#12345<<<3",
        )
        self.assertEqual(
            encoded["text/palindrome"],
            "::=~text='racecar'",
        )

    def test_aggregate_keeps_all_lanes(self) -> None:
        row = {
            "task_id": "fixture",
            "category": "scalar",
            "kern_native_16k": 40,
            "kern_native_exact_roundtrip": True,
            "kern_contract_ast": True,
        }
        for index, name in enumerate(
            (
                "python",
                "kern",
                "python_minifier",
                "gnu_apl",
                "cjam",
                "kona",
            ),
            start=1,
        ):
            row[f"{name}_cl100k"] = index * 10
            row[f"{name}_o200k"] = index * 11
            row[f"{name}_bytes"] = index * 12
            row[f"{name}_oracle_ok"] = True

        result = aggregate([row])

        self.assertEqual(result["programs"], 1)
        self.assertEqual(result["cl100k_base"]["kern"], 20)
        self.assertEqual(result["cl100k_base"]["cjam"], 50)
        self.assertEqual(result["native_system"]["kern_native_16k"], 40)
        self.assertEqual(result["functional"]["kona"], 1)
        self.assertEqual(result["structural"]["kern_contract_ast"], 1)

    def test_prior_screens_use_the_identical_registry(self) -> None:
        current = {
            "cl100k_base": {
                "kern": 91,
                "cjam": 93,
                "kona": 115,
                "gnu_apl": 142,
            }
        }

        market = cross_screen_market(frontier_pairs(), current)

        self.assertEqual(market["kern_cl100k_base_current"], 91)
        self.assertEqual(market["competitor_cl100k_base"]["jelly"], 128)
        self.assertEqual(market["competitor_cl100k_base"]["bqn"], 230)
        self.assertTrue(
            market["prior_screen_evidence"]["golf-languages"][
                "registry_match"
            ]
        )


if __name__ == "__main__":
    unittest.main()
