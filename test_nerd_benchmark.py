from __future__ import annotations

import ast
import unittest

from benchmark_modern import normalize_ast
from benchmark_nerd import (
    EXPECTED_PAIRS,
    PairResult,
    aggregate,
    normalize_stdout,
    paired_programs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class NerdBenchmarkTests(unittest.TestCase):
    def test_registry_contains_all_public_local_examples(self) -> None:
        pairs = paired_programs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(
            {pair.nerd_example for pair in pairs},
            {
                "calculator.nerd",
                "conditionals.nerd",
                "fizzbuzz.nerd",
                "functions.nerd",
                "loops.nerd",
                "math.nerd",
                "output.nerd",
            },
        )
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.nerd.strip())
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in paired_programs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_stdout_normalizes_only_numeric_lines(self) -> None:
        self.assertEqual(
            normalize_stdout("5.0\n-1.2500\nFizz\n"),
            "5\n-1.25\nFizz",
        )

    def test_aggregate_keeps_full_denominator(self) -> None:
        fixture = PairResult(
            task_id="fixture/0",
            nerd_example="fixture.nerd",
            python_sha256="a",
            nerd_sha256="b",
            python_cl100k=100,
            kern_cl100k=50,
            python_minifier_cl100k=70,
            nerd_cl100k=80,
            python_o200k=100,
            kern_o200k=55,
            python_minifier_o200k=75,
            nerd_o200k=85,
            kern_native_16k=40,
            nerd_lexer_tokens=60,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            nerd_parse_ok=True,
            nerd_compile_run_ok=True,
            nerd_oracle_ok=True,
            nerd_error_stage="",
            nerd_error_message="",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern_compact"], 50)
        self.assertEqual(row["cl100k_base"]["nerd"], 80)
        self.assertEqual(
            row["comparisons"]["kern_cl100k_below_nerd_pct"],
            37.5,
        )
        self.assertEqual(row["functional"]["nerd_oracle"], 1)


if __name__ == "__main__":
    unittest.main()
