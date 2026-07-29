from __future__ import annotations

import ast
import unittest

import tiktoken

from benchmark_market import (
    EncodedArtifact,
    MarketAdapter,
    aggregate,
    evaluate_case,
)
from benchmark_modern import Task


ENCODINGS = {
    name: tiktoken.get_encoding(name)
    for name in ("cl100k_base", "o200k_base")
}


class MarketBenchmarkTests(unittest.TestCase):
    def test_identity_adapter_preserves_full_denominator(self) -> None:
        source = "def add(a, b):\n    return a + b\n"
        task = Task("fixture", "fixture/0", "add", source)
        adapter = MarketAdapter(
            name="identity",
            version="test",
            encode=lambda value: EncodedArtifact(value),
            decode_to_python=lambda value: value,
            expected_ast_source=lambda value: value,
        )

        result = evaluate_case(task, adapter, ENCODINGS)
        rows = aggregate([result], ["identity"])

        self.assertTrue(result.ast_equal)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["total_cases"] == 1 for row in rows))
        self.assertTrue(all(row["token_cases"] == 1 for row in rows))
        self.assertTrue(all(row["parse_ok"] == 1 for row in rows))

    def test_encoded_tokens_survive_decode_failure(self) -> None:
        source = "def answer():\n    return 42\n"
        task = Task("fixture", "fixture/1", "answer", source)

        def fail_decode(_: str) -> str:
            raise ValueError("unsupported")

        adapter = MarketAdapter(
            name="broken_decoder",
            version="test",
            encode=lambda _: EncodedArtifact("@answer=42", 50.0),
            decode_to_python=fail_decode,
            expected_ast_source=lambda value: value,
        )

        result = evaluate_case(task, adapter, ENCODINGS)
        row = aggregate([result], ["broken_decoder"])[0]

        self.assertTrue(result.encode_ok)
        self.assertFalse(result.decode_ok)
        self.assertEqual(result.error_stage, "decode")
        self.assertEqual(row["token_cases"], 1)
        self.assertGreater(row["representation_tokens"], 0)
        self.assertEqual(row["parse_ok"], 0)
        self.assertEqual(row["full_conversion"], 0)

    def test_expected_semantic_ast_can_differ_from_source(self) -> None:
        source = "def choose():\n    return None\n"
        expected = "def choose():\n    return\n"
        task = Task("fixture", "fixture/2", "choose", source)
        adapter = MarketAdapter(
            name="semantic",
            version="test",
            encode=lambda _: EncodedArtifact("choose=>"),
            decode_to_python=lambda _: expected,
            expected_ast_source=lambda _: expected,
        )

        result = evaluate_case(task, adapter, ENCODINGS)

        self.assertTrue(result.parse_ok)
        self.assertTrue(result.ast_equal)
        ast.parse(result.decoded)


if __name__ == "__main__":
    unittest.main()
