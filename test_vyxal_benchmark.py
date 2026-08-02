from __future__ import annotations

import ast
import unittest

from benchmark_compact_languages import normalize_ast
from benchmark_vyxal import (
    EXPECTED_PAIRS,
    VYXAL_CODEPAGE,
    VyxalResult,
    aggregate,
    encode_vyxal_codepage,
    vyxal_pairs,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


class VyxalBenchmarkTests(unittest.TestCase):
    def test_registry_is_fixed_complete_and_codepage_valid(self) -> None:
        pairs = vyxal_pairs()

        self.assertEqual(len(pairs), EXPECTED_PAIRS)
        self.assertEqual(len({pair.task_id for pair in pairs}), EXPECTED_PAIRS)
        self.assertEqual(len(VYXAL_CODEPAGE), 256)
        for pair in pairs:
            with self.subTest(task_id=pair.task_id):
                ast.parse(pair.python)
                encoded = encode_vyxal_codepage(pair.vyxal)
                self.assertEqual(len(encoded), len(pair.vyxal))
                self.assertEqual(
                    "".join(VYXAL_CODEPAGE[value] for value in encoded),
                    pair.vyxal,
                )

    def test_every_python_pair_roundtrips_through_kern_contract(self) -> None:
        for pair in vyxal_pairs():
            with self.subTest(task_id=pair.task_id):
                tree = ast.parse(pair.python)
                encoded = transpile(pair.python, compact=True)
                decoded = compile_kern(encoded)
                expected = ast.unparse(compact_tree(tree))

                self.assertEqual(normalize_ast(decoded), normalize_ast(expected))

    def test_non_codepage_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_vyxal_codepage("🙂")

    def test_aggregate_keeps_token_and_codepage_lanes_separate(self) -> None:
        fixture = VyxalResult(
            task_id="fixture/0",
            category="scalar",
            python_sha256="a",
            kern_sha256="b",
            python_minifier_sha256="c",
            vyxal_sha256="d",
            vyxal_codepage_sha256="e",
            python_bytes=100,
            kern_bytes=50,
            python_minifier_bytes=70,
            vyxal_bytes=60,
            vyxal_codepage_units=20,
            python_cl100k=100,
            kern_cl100k=40,
            python_minifier_cl100k=70,
            vyxal_cl100k=50,
            python_o200k=100,
            kern_o200k=42,
            python_minifier_o200k=72,
            vyxal_o200k=52,
            kern_native_16k=30,
            kern_native_exact_roundtrip=True,
            kern_contract_ast=True,
            vyxal_codepage_roundtrip=True,
            python_oracle_ok=True,
            kern_oracle_ok=True,
            python_minifier_oracle_ok=True,
            vyxal_oracle_ok=True,
            vyxal_error="",
        )

        row = aggregate([fixture])

        self.assertEqual(row["programs"], 1)
        self.assertEqual(row["cl100k_base"]["kern"], 40)
        self.assertEqual(row["cl100k_base"]["vyxal"], 50)
        self.assertEqual(row["vyxal_codepage_units"], 20)
        self.assertEqual(row["comparisons"]["shared_kern_wins"], 1)
        self.assertEqual(row["comparisons"]["shared_ties"], 0)
        self.assertEqual(row["comparisons"]["shared_vyxal_wins"], 0)
        self.assertEqual(row["functional"]["vyxal"], 1)


if __name__ == "__main__":
    unittest.main()
