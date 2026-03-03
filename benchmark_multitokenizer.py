"""
benchmark_multitokenizer.py

Paso 8 roadmap:
- Medir reduccion de tokens Python vs Kern en:
  - HumanEval (164)
  - MBPP train (374)
- Tokenizers:
  - cl100k_base
  - o200k_base
  - LLaMA (TinyLlama tokenizer)
  - CodeGen (Salesforce/codegen-350M-mono tokenizer)
"""

from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import tiktoken
from datasets import load_dataset
from human_eval.data import read_problems
from transformers import AutoTokenizer

from kern_compiler import compile_kern
from kern_transpiler import transpile


UNSUPPORTED_STMT = "# UNSUPPORTED:"
UNSUPPORTED_EXPR_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")


@dataclass
class TokenStat:
    python_tokens: int
    kern_tokens: int
    saved_tokens: int
    saved_pct: float


@dataclass
class CaseResult:
    dataset: str
    task_id: str
    transpile_ok: bool
    compile_ok: bool
    parse_back_ok: bool
    error_stage: str
    error_message: str
    token_stats: dict[str, TokenStat]


def load_humaneval() -> list[tuple[str, str]]:
    problems = read_problems()
    rows: list[tuple[str, str]] = []
    for task_id, p in problems.items():
        source = p["prompt"] + p["canonical_solution"]
        rows.append((task_id, source))
    return rows


def load_mbpp_train() -> list[tuple[str, str]]:
    ds = load_dataset("mbpp", split="train")
    rows: list[tuple[str, str]] = []
    for item in ds:
        rows.append((f"MBPP/{item['task_id']}", item["code"]))
    return rows


def build_tokenizers() -> dict[str, Callable[[str], int]]:
    enc_cl = tiktoken.get_encoding("cl100k_base")
    enc_o2 = tiktoken.get_encoding("o200k_base")

    llama_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    codegen_tok = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    return {
        "cl100k_base": lambda s: len(enc_cl.encode(s)),
        "o200k_base": lambda s: len(enc_o2.encode(s)),
        "llama_tinyllama": lambda s: len(llama_tok.encode(s, add_special_tokens=False)),
        "codegen_350m_mono": lambda s: len(codegen_tok.encode(s, add_special_tokens=False)),
    }


def evaluate_case(
    dataset: str,
    task_id: str,
    source: str,
    tokenizers: dict[str, Callable[[str], int]],
) -> CaseResult:
    try:
        kern = transpile(source)
        if UNSUPPORTED_STMT in kern or UNSUPPORTED_EXPR_RE.search(kern):
            return CaseResult(
                dataset=dataset,
                task_id=task_id,
                transpile_ok=False,
                compile_ok=False,
                parse_back_ok=False,
                error_stage="transpile",
                error_message="unsupported marker emitted",
                token_stats={},
            )
    except Exception as exc:
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            transpile_ok=False,
            compile_ok=False,
            parse_back_ok=False,
            error_stage="transpile",
            error_message=str(exc),
            token_stats={},
        )

    try:
        py_back = compile_kern(kern)
    except Exception as exc:
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            transpile_ok=True,
            compile_ok=False,
            parse_back_ok=False,
            error_stage="compile",
            error_message=str(exc),
            token_stats={},
        )

    try:
        ast.parse(py_back)
    except Exception as exc:
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            transpile_ok=True,
            compile_ok=True,
            parse_back_ok=False,
            error_stage="parse_back",
            error_message=str(exc),
            token_stats={},
        )

    stats: dict[str, TokenStat] = {}
    for name, tok_len in tokenizers.items():
        py_t = tok_len(source)
        ke_t = tok_len(kern)
        saved = py_t - ke_t
        pct = (saved / py_t) * 100 if py_t else 0.0
        stats[name] = TokenStat(
            python_tokens=py_t,
            kern_tokens=ke_t,
            saved_tokens=saved,
            saved_pct=pct,
        )

    return CaseResult(
        dataset=dataset,
        task_id=task_id,
        transpile_ok=True,
        compile_ok=True,
        parse_back_ok=True,
        error_stage="",
        error_message="",
        token_stats=stats,
    )


def aggregate_summary(
    results: list[CaseResult],
    tokenizers: list[str],
    datasets: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for ds_name in datasets + ["overall"]:
        if ds_name == "overall":
            subset = results
        else:
            subset = [r for r in results if r.dataset == ds_name]

        total_cases = len(subset)
        valid_cases = [r for r in subset if r.parse_back_ok]
        valid_count = len(valid_cases)

        for tok in tokenizers:
            py_total = sum(r.token_stats[tok].python_tokens for r in valid_cases)
            ke_total = sum(r.token_stats[tok].kern_tokens for r in valid_cases)
            saved = py_total - ke_total
            pct = (saved / py_total) * 100 if py_total else 0.0
            rows.append(
                {
                    "dataset": ds_name,
                    "tokenizer": tok,
                    "total_cases": total_cases,
                    "valid_cases": valid_count,
                    "python_tokens": py_total,
                    "kern_tokens": ke_total,
                    "saved_tokens": saved,
                    "saved_pct": round(pct, 2),
                }
            )
    return rows


def main() -> None:
    tokenizers = build_tokenizers()
    tokenizers_order = list(tokenizers.keys())

    humaneval_rows = load_humaneval()
    mbpp_rows = load_mbpp_train()

    all_inputs: list[tuple[str, str, str]] = []
    all_inputs.extend([("humaneval", task_id, src) for task_id, src in humaneval_rows])
    all_inputs.extend([("mbpp_train", task_id, src) for task_id, src in mbpp_rows])

    results: list[CaseResult] = []
    total = len(all_inputs)
    print(f"Benchmarking {total} casos ({len(humaneval_rows)} HumanEval + {len(mbpp_rows)} MBPP)...")

    for i, (ds, task_id, src) in enumerate(all_inputs, start=1):
        results.append(evaluate_case(ds, task_id, src, tokenizers))
        if i % 50 == 0 or i == total:
            print(f"  progreso: {i}/{total}")

    summary = aggregate_summary(
        results=results,
        tokenizers=tokenizers_order,
        datasets=["humaneval", "mbpp_train"],
    )

    fail = [r for r in results if not r.parse_back_ok]
    print("\nResumen de conversion:")
    print(f"  total casos: {len(results)}")
    print(f"  conversion valida: {len(results) - len(fail)}/{len(results)}")
    if fail:
        print("  primeros fallos:")
        for r in fail[:10]:
            print(f"    - {r.dataset}:{r.task_id} stage={r.error_stage} msg={r.error_message}")

    print("\nTabla resumida:")
    for row in summary:
        if row["dataset"] == "overall":
            mark = "TOTAL"
        elif row["dataset"] == "humaneval":
            mark = "HumanEval"
        else:
            mark = "MBPP"
        print(
            f"  {mark:<10} | {row['tokenizer']:<18} | "
            f"{row['python_tokens']:>8} -> {row['kern_tokens']:<8} | {row['saved_pct']:>6.2f}%"
        )

    details_path = Path(__file__).with_name("benchmark_multitokenizer_details.json")
    summary_json_path = Path(__file__).with_name("benchmark_multitokenizer_summary.json")
    summary_csv_path = Path(__file__).with_name("benchmark_multitokenizer_summary.csv")

    details_payload = []
    for r in results:
        raw = asdict(r)
        # Flatten dataclass objects under token_stats for JSON readability.
        raw["token_stats"] = {
            name: asdict(stat) for name, stat in r.token_stats.items()
        }
        details_payload.append(raw)

    details_path.write_text(json.dumps(details_payload, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "tokenizer",
                "total_cases",
                "valid_cases",
                "python_tokens",
                "kern_tokens",
                "saved_tokens",
                "saved_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nReporte detallado: {details_path}")
    print(f"Resumen JSON:      {summary_json_path}")
    print(f"Resumen CSV:       {summary_csv_path}")


if __name__ == "__main__":
    main()
