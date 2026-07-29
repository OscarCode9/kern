from __future__ import annotations

import ast
import unittest

from benchmark_compact_languages import normalize_ast, normalize_stdout
from benchmark_array_languages import (
    EXPECTED_PAIRS,
    ArrayResult,
    aggregate,
    array_pairs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class ArrayLanguagesBenchmarkTests(unittest.TestCase):
    def test_registry_is_fixed_complete_and_valid(self) -> None:
        pairs = array_pairs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(
            {pair.category for pair in pairs},
            {"array", "recurrence", "reduction", "scalar", "text"},
        )
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.uiua.strip())
                self.assertTrue(pair.bqn.strip())
                self.assertIn(pair.bqn_mode, {"e", "o", "p"})
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in array_pairs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_stdout_normalization_does_not_strip_punctuation(self) -> None:
        self.assertEqual(normalize_stdout("[1, 2]\n"), "[1, 2]")

    def test_aggregate_keeps_full_denominator_and_system_lanes(self) -> None:
        fixture = ArrayResult(
            task_id="fixture/0",
            category="scalar",
            python_sha256="a",
            kern_sha256="b",
            python_minifier_sha256="c",
            uiua_sha256="d",
            bqn_sha256="e",
            python_bytes=100,
            kern_bytes=50,
            python_minifier_bytes=70,
            uiua_bytes=30,
            bqn_bytes=25,
            python_cl100k=100,
            kern_cl100k=50,
            python_minifier_cl100k=70,
            uiua_cl100k=30,
            bqn_cl100k=25,
            python_o200k=100,
            kern_o200k=55,
            python_minifier_o200k=75,
            uiua_o200k=32,
            bqn_o200k=22,
            kern_native_16k=40,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            uiua_oracle_ok=True,
            bqn_oracle_ok=True,
            uiua_error="",
            bqn_error="",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern"], 50)
        self.assertEqual(row["cl100k_base"]["uiua"], 30)
        self.assertEqual(row["native_system"]["kern_native_16k"], 40)
        self.assertEqual(
            row["comparisons"]["shared_kern_below_pct"]["bqn"],
            -100.0,
        )
        self.assertEqual(row["functional"]["bqn"], 1)
        self.assertEqual(row["categories"]["scalar"]["programs"], 1)


if __name__ == "__main__":
    unittest.main()
