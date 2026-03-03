"""
benchmark_humaneval_functional.py

Paso 7 (validacion funcional):
- Python -> Kern -> Python
- Ejecuta tests reales de HumanEval con check(entry_point)
- Reporta pass rate funcional y ahorro de tokens
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from human_eval.data import read_problems
from human_eval.execution import check_correctness

from kern_compiler import compile_kern
from kern_transpiler import transpile


@dataclass
class FunctionalResult:
    task_id: str
    transpile_ok: bool
    compile_ok: bool
    functional_passed: bool
    stage: str
    message: str
    py_tokens: int
    kern_tokens: int
    token_saved: int
    token_saved_pct: float


def run_case(task_id: str, problem: dict, enc, timeout: float) -> FunctionalResult:
    source = problem["prompt"] + problem["canonical_solution"]
    py_tokens = len(enc.encode(source))

    try:
        kern = transpile(source)
    except Exception as exc:
        return FunctionalResult(
            task_id=task_id,
            transpile_ok=False,
            compile_ok=False,
            functional_passed=False,
            stage="transpile",
            message=str(exc),
            py_tokens=py_tokens,
            kern_tokens=0,
            token_saved=0,
            token_saved_pct=0.0,
        )

    kern_tokens = len(enc.encode(kern))
    token_saved = py_tokens - kern_tokens
    token_saved_pct = (token_saved / py_tokens * 100) if py_tokens else 0.0

    try:
        py_back = compile_kern(kern)
    except Exception as exc:
        return FunctionalResult(
            task_id=task_id,
            transpile_ok=True,
            compile_ok=False,
            functional_passed=False,
            stage="compile",
            message=str(exc),
            py_tokens=py_tokens,
            kern_tokens=kern_tokens,
            token_saved=token_saved,
            token_saved_pct=token_saved_pct,
        )

    # Run check(entry_point) on the fully reconstructed Python.
    # We set prompt="" so check_correctness executes exactly `py_back` + tests.
    eval_problem = {
        "task_id": problem["task_id"],
        "prompt": "",
        "test": problem["test"],
        "entry_point": problem["entry_point"],
    }
    result = check_correctness(eval_problem, py_back, timeout=timeout)
    passed = bool(result["passed"])

    return FunctionalResult(
        task_id=task_id,
        transpile_ok=True,
        compile_ok=True,
        functional_passed=passed,
        stage="" if passed else "functional",
        message="" if passed else str(result.get("result", "failed")),
        py_tokens=py_tokens,
        kern_tokens=kern_tokens,
        token_saved=token_saved,
        token_saved_pct=token_saved_pct,
    )


def baseline_case(task_id: str, problem: dict, timeout: float) -> tuple[bool, str]:
    result = check_correctness(problem, problem["canonical_solution"], timeout=timeout)
    return bool(result["passed"]), str(result.get("result", ""))


def main() -> None:
    timeout = 3.0
    problems = read_problems()
    enc = tiktoken.get_encoding("cl100k_base")

    # Baseline sanity-check (canonical solution should pass).
    baseline_pass = 0
    baseline_failures: list[tuple[str, str]] = []
    for task_id, problem in problems.items():
        ok, msg = baseline_case(task_id, problem, timeout=timeout)
        if ok:
            baseline_pass += 1
        else:
            baseline_failures.append((task_id, msg))

    results: list[FunctionalResult] = []
    for task_id, problem in problems.items():
        results.append(run_case(task_id, problem, enc, timeout=timeout))

    total = len(results)
    transpile_ok = sum(r.transpile_ok for r in results)
    compile_ok = sum(r.compile_ok for r in results)
    functional_ok = sum(r.functional_passed for r in results)

    trans_ok_rows = [r for r in results if r.transpile_ok]
    total_py = sum(r.py_tokens for r in trans_ok_rows)
    total_kern = sum(r.kern_tokens for r in trans_ok_rows)
    total_saved = total_py - total_kern
    total_saved_pct = (total_saved / total_py * 100) if total_py else 0.0

    print("\n" + "=" * 76)
    print("HUMANEVAL FUNCTIONAL BENCHMARK (Python -> Kern -> Python -> check())")
    print("=" * 76)
    print(f"Total casos:           {total}")
    print(f"Baseline canonical:    {baseline_pass}/{total} ({100*baseline_pass/total:.1f}%)")
    print(f"Transpile OK:          {transpile_ok}/{total} ({100*transpile_ok/total:.1f}%)")
    print(f"Compile OK:            {compile_ok}/{total} ({100*compile_ok/total:.1f}%)")
    print(f"Functional pass:       {functional_ok}/{total} ({100*functional_ok/total:.1f}%)")
    print("-" * 76)
    print(f"Tokens Python:         {total_py}")
    print(f"Tokens Kern:           {total_kern}")
    print(f"Ahorro total:          {total_saved} ({total_saved_pct:.1f}%)")
    print("=" * 76)

    if baseline_failures:
        print("\nBaseline failures (canonical) [max 10]:")
        for task_id, msg in baseline_failures[:10]:
            print(f"  - {task_id:<14} {msg}")

    failures = [r for r in results if not r.functional_passed]
    if failures:
        print("\nRound-trip functional failures [max 15]:")
        for r in failures[:15]:
            print(f"  - {r.task_id:<14} stage={r.stage:<10} msg={r.message}")

    out_path = Path(__file__).with_name("humaneval_functional_report.json")
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"\nReporte detallado: {out_path}")


if __name__ == "__main__":
    main()
