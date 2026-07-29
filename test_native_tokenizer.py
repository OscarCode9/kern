from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer

from benchmark_native_tokenizer import (
    NativeResult,
    aggregate_modern,
    aggregate_pairs,
)
from train_kern_tokenizer import (
    PRETOKENIZER_CANDIDATES,
    aggregate_hash,
    create_tokenizer,
    train_candidate,
)


class NativeTokenizerTests(unittest.TestCase):
    fixtures = (
        "@add(a,b)=a+b",
        "@choose(x){?x>1:^x;:0}",
        '@unicode(x)="á🙂"+x',
        "@loop(xs){~x:xs{!x};^#}",
    )

    def test_aggregate_hash_is_length_delimited(self) -> None:
        self.assertNotEqual(
            aggregate_hash(("ab", "c")),
            aggregate_hash(("a", "bc")),
        )

    def test_all_candidates_are_lossless(self) -> None:
        for candidate in PRETOKENIZER_CANDIDATES:
            with self.subTest(candidate=candidate):
                tokenizer = train_candidate(
                    candidate=candidate,
                    sources=self.fixtures,
                    vocab_size=257,
                )
                decoded = [
                    tokenizer.decode(tokenizer.encode(value).ids)
                    for value in self.fixtures
                ]
                self.assertEqual(decoded, list(self.fixtures))

    def test_saved_tokenizer_preserves_exact_source(self) -> None:
        tokenizer = create_tokenizer("bytelevel_regex")
        tokenizer = train_candidate(
            candidate="bytelevel_regex",
            sources=self.fixtures,
            vocab_size=257,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(str(path))
            loaded = Tokenizer.from_file(str(path))
            for fixture in self.fixtures:
                self.assertEqual(
                    loaded.decode(loaded.encode(fixture).ids),
                    fixture,
                )

    def test_native_aggregates_keep_full_denominator(self) -> None:
        modern = NativeResult(
            dataset="HumanEval+",
            task_id="fixture/0",
            python_sha256="a",
            python_cl100k=100,
            kern_cl100k=70,
            kern_native_16k=50,
            python_minifier_cl100k=75,
            kern_native_exact_roundtrip=True,
        )
        pair = NativeResult(
            dataset="Toke public pairs",
            task_id="fixture/1",
            python_sha256="b",
            python_cl100k=100,
            kern_cl100k=80,
            kern_native_16k=40,
            python_minifier_cl100k=75,
            kern_native_exact_roundtrip=True,
            toke_sha256="c",
            toke_cl100k=150,
            toke_native_16k=80,
        )

        modern_rows = aggregate_modern([modern, pair])
        paired = aggregate_pairs([modern, pair])

        self.assertEqual(modern_rows[-1]["programs"], 1)
        self.assertEqual(
            modern_rows[-1]["kern_native_saved_vs_python_cl100k_pct"],
            50.0,
        )
        self.assertEqual(paired["programs"], 1)
        self.assertEqual(paired["kern_native_below_toke_native_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
