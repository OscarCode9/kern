"""
test_baseline_adapters.py

Lightweight sanity tests for representation adapters used in benchmark_head_to_head.py.
Run directly:
    python3 test_baseline_adapters.py
"""

from __future__ import annotations

import ast

from human_eval.data import read_problems

from benchmark_head_to_head import (
    kern_adapter,
    python_adapter,
    simpy_adapter,
    token_sugar_adapter,
)


def _sample_sources(n: int = 20) -> list[str]:
    problems = read_problems()
    rows = []
    for _, p in list(problems.items())[:n]:
        rows.append(p["prompt"] + p["canonical_solution"])
    return rows


def test_python_adapter() -> None:
    adapter = python_adapter()
    src = "def add(a,b):\n    return a+b\n"
    encoded = adapter.encode(src)
    decoded = adapter.decode_to_python(encoded)
    assert decoded == src
    ast.parse(decoded)


def test_kern_adapter() -> None:
    adapter = kern_adapter()
    src = "def add(a,b):\n    return a+b\n"
    encoded = adapter.encode(src)
    decoded = adapter.decode_to_python(encoded)
    ast.parse(decoded)


def test_simpy_adapter_humaneval_sample() -> None:
    adapter = simpy_adapter("external/SimPy")
    ok = 0
    for src in _sample_sources(20):
        encoded = adapter.encode(src)
        decoded = adapter.decode_to_python(encoded)
        ast.parse(decoded)
        ok += 1
    assert ok == 20


def test_token_sugar_adapter_humaneval_sample() -> None:
    adapter = token_sugar_adapter(
        "external/TokenSugar",
        "external/TokenSugar/mined_sugars.json",
    )
    ok = 0
    for src in _sample_sources(20):
        encoded = adapter.encode(src)
        decoded = adapter.decode_to_python(encoded)
        ast.parse(decoded)
        ok += 1
    assert ok == 20


if __name__ == "__main__":
    test_python_adapter()
    test_kern_adapter()
    test_simpy_adapter_humaneval_sample()
    test_token_sugar_adapter_humaneval_sample()
    print("All baseline adapter tests passed.")
