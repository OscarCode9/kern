from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from benchmark_toke import (
    PairResult,
    aggregate_tokens,
    build_python_programs,
    first_error_code,
    robustness_summary,
)


class TokeBenchmarkTests(unittest.TestCase):
    def test_extracts_task_without_registry_scaffolding(self) -> None:
        source = '''
import math

SOLUTIONS = {}

def task(task_id):
    def decorator(fn):
        SOLUTIONS[task_id] = fn
        return fn
    return decorator

def helper(value):
    return value + 1

@task("task-a-0001")
def solve(value: int) -> int:
    return math.floor(helper(value))
'''
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "solutions.py"
            path.write_text(source, encoding="utf-8")
            programs = build_python_programs(path)

        program = programs["task-a-0001"]
        ast.parse(program)
        self.assertIn("import json, sys", program)
        self.assertIn("import math", program)
        self.assertIn("def helper", program)
        self.assertIn("def solve", program)
        self.assertNotIn("SOLUTIONS", program)
        self.assertNotIn("def task", program)
        self.assertNotIn("@task", program)

    def test_first_error_ignores_warning(self) -> None:
        output = "\n".join(
            [
                '{"severity":"warning","error_code":"W1020"}',
                '{"severity":"error","error_code":"E2002"}',
            ]
        )
        self.assertEqual(first_error_code(output), "E2002")

    def test_aggregate_keeps_full_denominator(self) -> None:
        result = PairResult(
            task_id="task-a-0001",
            python_sha256="a",
            toke_sha256="b",
            python_cl100k=100,
            kern_compact_cl100k=70,
            python_minifier_cl100k=75,
            toke_cl100k=150,
            python_o200k=110,
            kern_compact_o200k=80,
            python_minifier_o200k=85,
            toke_o200k=160,
            toke_native_tokens=90,
            kern_decode_ok=True,
            kern_parse_ok=True,
            kern_contract_ast=True,
            toke_check_ok=False,
            toke_error_code="E2002",
        )

        rows = aggregate_tokens([result])

        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["programs"] == 1 for row in rows))
        toke_row = next(
            row
            for row in rows
            if row["tokenizer"] == "cl100k_base"
            and row["representation"] == "toke"
        )
        self.assertEqual(toke_row["representation_tokens"], 150)
        self.assertEqual(toke_row["saved_pct"], -50.0)
        robustness = robustness_summary([result])
        self.assertEqual(robustness["kern_shared_tokenizer_wins"], 1)
        self.assertEqual(
            robustness["excluding_largest_toke_source"]["programs"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
