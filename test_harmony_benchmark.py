from __future__ import annotations

import unittest

import tiktoken

from benchmark_harmony import aggregate, encoding_contract, score_case
from benchmark_head_to_head import build_tokenizers
from benchmark_modern import Task


class HarmonyBenchmarkTests(unittest.TestCase):
    def test_harmony_shares_the_ordinary_o200k_contract(self) -> None:
        contract = encoding_contract()

        self.assertTrue(contract["pattern_equal"])
        self.assertTrue(contract["mergeable_ranks_equal"])
        self.assertEqual(
            contract["mergeable_ranks_sha256"]["o200k_base"],
            contract["mergeable_ranks_sha256"]["o200k_harmony"],
        )
        self.assertGreater(
            contract["special_token_count"]["o200k_harmony"],
            contract["special_token_count"]["o200k_base"],
        )

    def test_source_ids_match_for_every_representation(self) -> None:
        task = Task(
            dataset="fixture",
            task_id="fixture/0",
            entry_point="square",
            source="def square(value):\n    return value * value\n",
        )
        encodings = {
            name: tiktoken.get_encoding(name)
            for name in ("cl100k_base", "o200k_base", "o200k_harmony")
        }
        rows = [
            score_case(task, representation, encodings)
            for representation in (
                "python",
                "kern",
                "kern_compact",
                "python_minifier",
            )
        ]

        result = aggregate(rows)

        self.assertEqual(result["programs"], 1)
        self.assertEqual(result["representation_rows"], 4)
        self.assertEqual(result["o200k_exact_id_matches"], 4)
        self.assertEqual(result["o200k_token_delta"], 0)
        self.assertEqual(result["max_absolute_case_delta"], 0)

    def test_head_to_head_accepts_harmony(self) -> None:
        counters = build_tokenizers(["o200k_base", "o200k_harmony"])
        source = "def add(left, right):\n    return left + right\n"

        self.assertEqual(
            counters["o200k_base"](source),
            counters["o200k_harmony"](source),
        )


if __name__ == "__main__":
    unittest.main()
