"""
benchmark_humaneval_roundtrip.py

Paso 7 roadmap:
- Python -> Kern -> Python sobre HumanEval
- Verifica parseo y equivalencia de AST (normalizada sin docstrings)
- Reporta reducción de tokens (cl100k_base)
"""

from __future__ import annotations

import ast
import json
import re
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import tiktoken
from human_eval.data import read_problems

from kern_compiler import compile_kern
from kern_transpiler import transpile


UNSUPPORTED_STMT = "# UNSUPPORTED:"
UNSUPPORTED_EXPR_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")


@dataclass
class CaseResult:
    task_id: str
    transpile_ok: bool
    compile_ok: bool
    parse_ok: bool
    ast_equal: bool
    py_tokens: int
    kern_tokens: int
    token_saved: int
    token_saved_pct: float
    error_stage: str
    error_message: str


class DocstringStripper(ast.NodeTransformer):
    """Remove no-op string Expr nodes to avoid false AST mismatches."""

    def _is_nop_string_expr(self, stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _strip_doc(self, body: list[ast.stmt]) -> list[ast.stmt]:
        return [stmt for stmt in body if not self._is_nop_string_expr(stmt)]

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        node.body = self._strip_doc(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):  # type: ignore[override]
        node.body = self._strip_doc(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):  # type: ignore[override]
        node.body = self._strip_doc(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):  # type: ignore[override]
        node.body = self._strip_doc(node.body)
        self.generic_visit(node)
        return node


def normalize_ast(source: str) -> str:
    tree = ast.parse(source)
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def to_source(problem: dict) -> str:
    # HumanEval format: prompt trae firma + docstring; canonical_solution trae cuerpo indentado.
    return problem["prompt"] + problem["canonical_solution"]


def run_case(task_id: str, source: str, enc) -> CaseResult:
    py_tokens = len(enc.encode(source))
    kern_tokens = 0

    try:
        kern = transpile(source)
        if UNSUPPORTED_STMT in kern or UNSUPPORTED_EXPR_RE.search(kern):
            return CaseResult(
                task_id=task_id,
                transpile_ok=False,
                compile_ok=False,
                parse_ok=False,
                ast_equal=False,
                py_tokens=py_tokens,
                kern_tokens=0,
                token_saved=0,
                token_saved_pct=0.0,
                error_stage="transpile",
                error_message="unsupported node marker in Kern output",
            )
    except Exception as exc:
        return CaseResult(
            task_id=task_id,
            transpile_ok=False,
            compile_ok=False,
            parse_ok=False,
            ast_equal=False,
            py_tokens=py_tokens,
            kern_tokens=0,
            token_saved=0,
            token_saved_pct=0.0,
            error_stage="transpile",
            error_message=str(exc),
        )

    try:
        py_back = compile_kern(kern)
    except Exception as exc:
        kern_tokens = len(enc.encode(kern))
        saved = py_tokens - kern_tokens
        pct = (saved / py_tokens) * 100 if py_tokens else 0.0
        return CaseResult(
            task_id=task_id,
            transpile_ok=True,
            compile_ok=False,
            parse_ok=False,
            ast_equal=False,
            py_tokens=py_tokens,
            kern_tokens=kern_tokens,
            token_saved=saved,
            token_saved_pct=pct,
            error_stage="compile",
            error_message=str(exc),
        )

    try:
        ast.parse(py_back)
    except Exception as exc:
        kern_tokens = len(enc.encode(kern))
        saved = py_tokens - kern_tokens
        pct = (saved / py_tokens) * 100 if py_tokens else 0.0
        return CaseResult(
            task_id=task_id,
            transpile_ok=True,
            compile_ok=True,
            parse_ok=False,
            ast_equal=False,
            py_tokens=py_tokens,
            kern_tokens=kern_tokens,
            token_saved=saved,
            token_saved_pct=pct,
            error_stage="parse_back",
            error_message=str(exc),
        )

    try:
        left = normalize_ast(source)
        right = normalize_ast(py_back)
        equal = left == right
    except Exception:
        equal = False
        err = traceback.format_exc(limit=1).strip().splitlines()[-1]
        kern_tokens = len(enc.encode(kern))
        saved = py_tokens - kern_tokens
        pct = (saved / py_tokens) * 100 if py_tokens else 0.0
        return CaseResult(
            task_id=task_id,
            transpile_ok=True,
            compile_ok=True,
            parse_ok=True,
            ast_equal=False,
            py_tokens=py_tokens,
            kern_tokens=kern_tokens,
            token_saved=saved,
            token_saved_pct=pct,
            error_stage="ast_compare",
            error_message=err,
        )

    kern_tokens = len(enc.encode(kern))
    saved = py_tokens - kern_tokens
    pct = (saved / py_tokens) * 100 if py_tokens else 0.0

    return CaseResult(
        task_id=task_id,
        transpile_ok=True,
        compile_ok=True,
        parse_ok=True,
        ast_equal=equal,
        py_tokens=py_tokens,
        kern_tokens=kern_tokens,
        token_saved=saved,
        token_saved_pct=pct,
        error_stage="",
        error_message="" if equal else "AST differs",
    )


def main() -> None:
    problems = read_problems()
    enc = tiktoken.get_encoding("cl100k_base")
    results: list[CaseResult] = []

    for task_id, problem in problems.items():
        source = to_source(problem)
        results.append(run_case(task_id, source, enc))

    total = len(results)
    transpile_ok = sum(r.transpile_ok for r in results)
    compile_ok = sum(r.compile_ok for r in results)
    parse_ok = sum(r.parse_ok for r in results)
    ast_equal = sum(r.ast_equal for r in results)

    ok_for_tokens = [r for r in results if r.transpile_ok]
    total_py = sum(r.py_tokens for r in ok_for_tokens)
    total_kern = sum(r.kern_tokens for r in ok_for_tokens)
    total_saved = total_py - total_kern
    avg_saved_pct = (total_saved / total_py) * 100 if total_py else 0.0

    print("\n" + "=" * 72)
    print("HUMANEVAL ROUND-TRIP BENCHMARK (Python -> Kern -> Python)")
    print("=" * 72)
    print(f"Total casos:          {total}")
    print(f"Transpile OK:         {transpile_ok}/{total} ({100*transpile_ok/total:.1f}%)")
    print(f"Compile OK:           {compile_ok}/{total} ({100*compile_ok/total:.1f}%)")
    print(f"Parse back OK:        {parse_ok}/{total} ({100*parse_ok/total:.1f}%)")
    print(f"AST igual (normaliz): {ast_equal}/{total} ({100*ast_equal/total:.1f}%)")
    print("-" * 72)
    print(f"Tokens Python:        {total_py}")
    print(f"Tokens Kern:          {total_kern}")
    print(f"Ahorro total:         {total_saved} ({avg_saved_pct:.1f}%)")
    print("=" * 72)

    failures = [r for r in results if not r.ast_equal]
    if failures:
        print("\nPrimeros fallos (max 15):")
        for r in failures[:15]:
            print(f"  - {r.task_id:<14} stage={r.error_stage:<10} msg={r.error_message}")

    out_path = Path(__file__).with_name("humaneval_roundtrip_report.json")
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"\nReporte detallado: {out_path}")


if __name__ == "__main__":
    main()
