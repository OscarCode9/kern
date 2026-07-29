from __future__ import annotations

import ast
import unittest

from benchmark_compact_languages import (
    EXPECTED_PAIRS,
    PairResult,
    aggregate,
    normalize_ast,
    normalize_stdout,
    paired_programs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class CompactLanguagesBenchmarkTests(unittest.TestCase):
    def test_registry_is_fixed_complete_and_valid(self) -> None:
        pairs = paired_programs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(
            {pair.category for pair in pairs},
            {"array", "recurrence", "reduction", "scalar", "text"},
        )
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.k.strip())
                self.assertTrue(pair.golfscript.strip())
                self.assertTrue(pair.j.strip())
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in paired_programs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_stdout_normalization_preserves_value_order(self) -> None:
        self.assertEqual(
            normalize_stdout("  1   2\n3\tFizz  "),
            "1 2 3 Fizz",
        )

    def test_aggregate_keeps_full_denominator(self) -> None:
        fixture = PairResult(
            task_id="fixture/0",
            category="scalar",
            python_sha256="a",
            k_sha256="b",
            golfscript_sha256="c",
            j_sha256="d",
            python_bytes=100,
            kern_bytes=50,
            python_minifier_bytes=70,
            k_bytes=30,
            golfscript_bytes=35,
            j_bytes=40,
            python_cl100k=100,
            kern_cl100k=50,
            python_minifier_cl100k=70,
            k_cl100k=30,
            golfscript_cl100k=35,
            j_cl100k=40,
            python_o200k=100,
            kern_o200k=55,
            python_minifier_o200k=75,
            k_o200k=32,
            golfscript_o200k=37,
            j_o200k=42,
            kern_native_16k=40,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            k_oracle_ok=True,
            golfscript_oracle_ok=True,
            j_oracle_ok=True,
            k_error="",
            golfscript_error="",
            j_error="",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern"], 50)
        self.assertEqual(row["cl100k_base"]["k"], 30)
        self.assertEqual(
            row["comparisons"]["shared_kern_below_pct"]["k"],
            -66.66666666666666,
        )
        self.assertEqual(row["functional"]["j"], 1)
        self.assertEqual(row["categories"]["scalar"]["programs"], 1)


if __name__ == "__main__":
    unittest.main()
