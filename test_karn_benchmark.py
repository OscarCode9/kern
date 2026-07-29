from __future__ import annotations

import ast
import unittest

from benchmark_karn import EXPECTED_PAIRS, PairResult, aggregate, paired_programs
from benchmark_modern import normalize_ast
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class KarnBenchmarkTests(unittest.TestCase):
    def test_pair_registry_is_fixed_and_python_is_valid(self) -> None:
        pairs = paired_programs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                self.assertTrue(pair.karn.strip())
                self.assertTrue(pair.expected_stdout)

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in paired_programs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_aggregate_keeps_every_representation(self) -> None:
        fixture = PairResult(
            task_id="fixture/0",
            origin="fixture",
            python_sha256="a",
            karn_sha256="b",
            python_cl100k=100,
            kern_cl100k=50,
            python_minifier_cl100k=70,
            karn_cl100k=80,
            python_o200k=100,
            kern_o200k=55,
            python_minifier_o200k=75,
            karn_o200k=85,
            kern_native_16k=40,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            karn_check_ok=True,
            karn_interpreter_oracle_ok=True,
            karn_python_codegen_ok=False,
            karn_python_codegen_oracle_ok=False,
            karn_error_stage="generated_python",
            karn_error_message="fixture",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern_compact"], 50)
        self.assertEqual(row["cl100k_base"]["karn"], 80)
        self.assertEqual(
            row["comparisons"]["kern_cl100k_below_karn_pct"],
            37.5,
        )
        self.assertEqual(row["functional"]["karn_python_codegen"], 0)


if __name__ == "__main__":
    unittest.main()
