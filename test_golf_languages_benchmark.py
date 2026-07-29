from __future__ import annotations

import ast
import unittest

from benchmark_compact_languages import normalize_ast, normalize_stdout
from benchmark_golf_languages import (
    EXPECTED_PAIRS,
    GolfResult,
    aggregate,
    golf_pairs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class GolfLanguagesBenchmarkTests(unittest.TestCase):
    def test_registry_is_fixed_complete_and_valid(self) -> None:
        pairs = golf_pairs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(
            {pair.category for pair in pairs},
            {"array", "recurrence", "reduction", "scalar", "text"},
        )
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.pyth.strip())
                self.assertTrue(pair.jelly.strip())
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in golf_pairs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_stdout_normalization_does_not_strip_punctuation(self) -> None:
        self.assertEqual(normalize_stdout("[1, 2]\n"), "[1, 2]")

    def test_aggregate_keeps_full_denominator_and_separate_code_page(self) -> None:
        fixture = GolfResult(
            task_id="fixture/0",
            category="scalar",
            python_sha256="a",
            kern_sha256="b",
            python_minifier_sha256="c",
            pyth_sha256="d",
            jelly_sha256="e",
            python_bytes=100,
            kern_bytes=50,
            python_minifier_bytes=70,
            pyth_bytes=30,
            jelly_bytes=25,
            jelly_code_page_units=20,
            python_cl100k=100,
            kern_cl100k=50,
            python_minifier_cl100k=70,
            pyth_cl100k=30,
            jelly_cl100k=25,
            python_o200k=100,
            kern_o200k=55,
            python_minifier_o200k=75,
            pyth_o200k=32,
            jelly_o200k=22,
            kern_native_16k=40,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            pyth_oracle_ok=True,
            jelly_oracle_ok=True,
            pyth_error="",
            jelly_error="",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern"], 50)
        self.assertEqual(row["cl100k_base"]["pyth"], 30)
        self.assertEqual(row["jelly_code_page_units"], 20)
        self.assertEqual(
            row["comparisons"]["shared_kern_below_pct"]["jelly"],
            -100.0,
        )
        self.assertEqual(row["functional"]["jelly"], 1)
        self.assertEqual(row["categories"]["scalar"]["programs"], 1)


if __name__ == "__main__":
    unittest.main()
