"""Reproducible shared-corpus market benchmark for compact code languages.

This harness is deliberately stricter than comparing headline percentages:

* every representation receives the same code-only Python source;
* token totals use the same production tokenizers and retain the full corpus
  denominator;
* encoded text is counted even when the competitor cannot decode it;
* decoded Python must parse before it can enter functional evaluation; and
* EvalPlus failures remain failures instead of disappearing from token totals.

Sigil is the first external language with a public Python-to-language converter
and language-to-Python compiler that can run this protocol end to end. Toke and
KARN require paired, independently verified implementations and are tracked in
``benchmark_results/market/competitors.json`` until those lanes are available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import python_minifier
import tiktoken

from benchmark_modern import (
    BIGCODEBENCH_REVISION,
    TOKENIZERS,
    Task,
    load_bigcodebench_tasks,
    load_evalplus_tasks,
    normalize_ast,
    write_grouped_bar_svg,
)
from evalplus.data import (
    get_human_eval_plus_hash,
    get_mbpp_plus_hash,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


DEFAULT_REPRESENTATIONS = (
    "python",
    "kern_compact",
    "python_minifier",
    "sigil",
)
SIGIL_VERSION = "0.1.0"
SIGIL_WHEEL_SHA256 = (
    "eba9b1a755c207d9db9dc658abf612c72f5a163aa72b597"
    "45ff9511ff8e7e31e"
)


@dataclass(frozen=True)
class EncodedArtifact:
    text: str
    conversion_percent: float = 100.0


@dataclass(frozen=True)
class MarketAdapter:
    name: str
    version: str
    encode: Callable[[str], EncodedArtifact]
    decode_to_python: Callable[[str], str]
    expected_ast_source: Callable[[str], str]
    requires_ast_equal: bool = False


@dataclass
class MarketCaseResult:
    dataset: str
    task_id: str
    representation: str
    encode_ok: bool
    conversion_percent: float | None
    decode_ok: bool
    parse_ok: bool
    ast_equal: bool | None
    python_cl100k: int
    representation_cl100k: int | None
    python_o200k: int
    representation_o200k: int | None
    encoded: str
    decoded: str
    error_stage: str
    error_message: str


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def error_text(exc: Exception, limit: int = 320) -> str:
    message = f"{type(exc).__name__}: {exc}".replace("\x00", "\\0")
    return message if len(message) <= limit else message[:limit] + "…"


def python_adapter() -> MarketAdapter:
    return MarketAdapter(
        name="python",
        version=sys.version.split()[0],
        encode=lambda source: EncodedArtifact(source),
        decode_to_python=lambda encoded: encoded,
        expected_ast_source=lambda source: source,
        requires_ast_equal=True,
    )


def kern_compact_adapter() -> MarketAdapter:
    return MarketAdapter(
        name="kern_compact",
        version="v0.4",
        encode=lambda source: EncodedArtifact(
            transpile(source, compact=True)
        ),
        decode_to_python=compile_kern,
        expected_ast_source=lambda source: ast.unparse(
            compact_tree(ast.parse(source))
        ),
        requires_ast_equal=True,
    )


def python_minifier_adapter() -> MarketAdapter:
    return MarketAdapter(
        name="python_minifier",
        version=package_version("python-minifier"),
        encode=lambda source: EncodedArtifact(
            python_minifier.minify(source, rename_globals=False)
        ),
        decode_to_python=lambda encoded: encoded,
        expected_ast_source=lambda source: source,
    )


def sigil_adapter() -> MarketAdapter:
    version = package_version("sigil-lang")
    if version != SIGIL_VERSION:
        raise RuntimeError(
            f"Market protocol requires sigil-lang=={SIGIL_VERSION}; "
            f"found {version}."
        )
    try:
        from src.converter.py_to_sigil import convert
        from src.sigil_sdk import transpile as transpile_sigil
    except ImportError as exc:
        raise RuntimeError(
            "Sigil is not installed. Install market dependencies with "
            "`pip install -r market-benchmark-requirements.txt`."
        ) from exc

    def encode(source: str) -> EncodedArtifact:
        result = convert(source)
        return EncodedArtifact(
            text=result.sigil,
            conversion_percent=float(result.percent_converted),
        )

    return MarketAdapter(
        name="sigil",
        version=version,
        encode=encode,
        decode_to_python=transpile_sigil,
        expected_ast_source=lambda source: source,
    )


def build_adapters(names: Iterable[str]) -> list[MarketAdapter]:
    factories: dict[str, Callable[[], MarketAdapter]] = {
        "python": python_adapter,
        "kern_compact": kern_compact_adapter,
        "python_minifier": python_minifier_adapter,
        "sigil": sigil_adapter,
    }
    return [factories[name]() for name in names]


def evaluate_case(
    task: Task,
    adapter: MarketAdapter,
    encodings: dict[str, Any],
) -> MarketCaseResult:
    source_counts = {
        name: len(encoding.encode(task.source))
        for name, encoding in encodings.items()
    }
    base = {
        "dataset": task.dataset,
        "task_id": task.task_id,
        "representation": adapter.name,
        "python_cl100k": source_counts["cl100k_base"],
        "python_o200k": source_counts["o200k_base"],
    }
    try:
        artifact = adapter.encode(task.source)
    except Exception as exc:
        return MarketCaseResult(
            **base,
            encode_ok=False,
            conversion_percent=None,
            decode_ok=False,
            parse_ok=False,
            ast_equal=None,
            representation_cl100k=None,
            representation_o200k=None,
            encoded="",
            decoded="",
            error_stage="encode",
            error_message=error_text(exc),
        )

    encoded_counts = {
        name: len(encoding.encode(artifact.text))
        for name, encoding in encodings.items()
    }
    encoded_fields = {
        "encode_ok": True,
        "conversion_percent": artifact.conversion_percent,
        "representation_cl100k": encoded_counts["cl100k_base"],
        "representation_o200k": encoded_counts["o200k_base"],
        "encoded": artifact.text,
    }
    try:
        decoded = adapter.decode_to_python(artifact.text)
    except Exception as exc:
        return MarketCaseResult(
            **base,
            **encoded_fields,
            decode_ok=False,
            parse_ok=False,
            ast_equal=None,
            decoded="",
            error_stage="decode",
            error_message=error_text(exc),
        )

    try:
        ast.parse(decoded)
    except Exception as exc:
        return MarketCaseResult(
            **base,
            **encoded_fields,
            decode_ok=True,
            parse_ok=False,
            ast_equal=None,
            decoded=decoded,
            error_stage="parse",
            error_message=error_text(exc),
        )

    try:
        expected = adapter.expected_ast_source(task.source)
        ast_equal = normalize_ast(expected) == normalize_ast(decoded)
    except Exception as exc:
        return MarketCaseResult(
            **base,
            **encoded_fields,
            decode_ok=True,
            parse_ok=True,
            ast_equal=False,
            decoded=decoded,
            error_stage="ast",
            error_message=error_text(exc),
        )

    return MarketCaseResult(
        **base,
        **encoded_fields,
        decode_ok=True,
        parse_ok=True,
        ast_equal=ast_equal,
        decoded=decoded,
        error_stage=(
            "ast"
            if adapter.requires_ast_equal and not ast_equal
            else ""
        ),
        error_message=(
            "normalized AST differs"
            if adapter.requires_ast_equal and not ast_equal
            else ""
        ),
    )


def aggregate(
    results: list[MarketCaseResult],
    representations: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted({result.dataset for result in results})
    for dataset in datasets:
        for representation in representations:
            subset = [
                result
                for result in results
                if result.dataset == dataset
                and result.representation == representation
            ]
            encoded = [result for result in subset if result.encode_ok]
            conversion_values = [
                result.conversion_percent
                for result in encoded
                if result.conversion_percent is not None
            ]
            for tokenizer in TOKENIZERS:
                source_attr = (
                    "python_cl100k"
                    if tokenizer == "cl100k_base"
                    else "python_o200k"
                )
                representation_attr = (
                    "representation_cl100k"
                    if tokenizer == "cl100k_base"
                    else "representation_o200k"
                )
                token_rows = [
                    result
                    for result in encoded
                    if getattr(result, representation_attr) is not None
                ]
                python_tokens = sum(
                    getattr(result, source_attr) for result in token_rows
                )
                representation_tokens = sum(
                    getattr(result, representation_attr)
                    for result in token_rows
                )
                saved_tokens = python_tokens - representation_tokens
                saved_pct = (
                    saved_tokens / python_tokens * 100
                    if python_tokens
                    else 0.0
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "tokenizer": tokenizer,
                        "total_cases": len(subset),
                        "token_cases": len(token_rows),
                        "encode_ok": len(encoded),
                        "full_conversion": sum(
                            value == 100.0 for value in conversion_values
                        ),
                        "mean_conversion_pct": round(
                            sum(conversion_values) / len(conversion_values),
                            4,
                        )
                        if conversion_values
                        else None,
                        "decode_ok": sum(
                            result.decode_ok for result in subset
                        ),
                        "parse_ok": sum(
                            result.parse_ok for result in subset
                        ),
                        "ast_equal": sum(
                            result.ast_equal is True for result in subset
                        ),
                        "python_tokens": python_tokens,
                        "representation_tokens": representation_tokens,
                        "saved_tokens": saved_tokens,
                        "saved_pct": round(saved_pct, 4),
                    }
                )
    return rows


def write_detail_csv(
    results: list[MarketCaseResult],
    path: Path,
) -> None:
    fields = [
        "dataset",
        "task_id",
        "representation",
        "encode_ok",
        "conversion_percent",
        "decode_ok",
        "parse_ok",
        "ast_equal",
        "python_cl100k",
        "representation_cl100k",
        "python_o200k",
        "representation_o200k",
        "error_stage",
        "error_message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            raw = asdict(result)
            writer.writerow({field: raw[field] for field in fields})


def write_evalplus_samples(
    tasks: Iterable[Task],
    results: list[MarketCaseResult],
    representation: str,
    dataset: str,
    path: Path,
) -> None:
    decoded = {
        (result.task_id, result.representation): result.decoded
        for result in results
        if result.parse_ok
    }
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            if task.dataset != dataset:
                continue
            solution = decoded.get((task.task_id, representation))
            if solution is None:
                solution = (
                    "raise RuntimeError("
                    "'representation conversion failed')"
                )
            handle.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "solution": solution,
                    }
                )
                + "\n"
            )


def run_evalplus(
    tasks: list[Task],
    results: list[MarketCaseResult],
    representations: Iterable[str],
    parallel: int,
    min_time_limit: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    functional: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(
        prefix="kern-market-evalplus-"
    ) as temp_dir:
        temp = Path(temp_dir)
        datasets = (("HumanEval+", "humaneval"), ("MBPP+", "mbpp"))
        for dataset, cli_dataset in datasets:
            functional[dataset] = {}
            for representation in representations:
                samples = temp / f"{cli_dataset}-{representation}.jsonl"
                write_evalplus_samples(
                    tasks,
                    results,
                    representation,
                    dataset,
                    samples,
                )
                command = [
                    sys.executable,
                    "-m",
                    "evalplus.evaluate",
                    "--dataset",
                    cli_dataset,
                    "--samples",
                    str(samples),
                    "--parallel",
                    str(parallel),
                    "--min_time_limit",
                    str(min_time_limit),
                ]
                environment = os.environ.copy()
                environment["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    env=environment,
                    text=True,
                )
                result_path = samples.with_name(
                    samples.stem + "_eval_results.json"
                )
                total = sum(
                    task.dataset == dataset for task in tasks
                )
                if completed.returncode != 0 or not result_path.exists():
                    functional[dataset][representation] = {
                        "total": total,
                        "base_pass": 0,
                        "plus_pass": 0,
                        "error": (
                            completed.stderr or completed.stdout
                        )[-2000:],
                    }
                    continue
                payload = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                evaluations = [
                    sample
                    for task_results in payload["eval"].values()
                    for sample in task_results
                ]
                functional[dataset][representation] = {
                    "total": len(evaluations),
                    "base_pass": sum(
                        sample["base_status"] == "pass"
                        for sample in evaluations
                    ),
                    "plus_pass": sum(
                        sample["base_status"] == "pass"
                        and sample["plus_status"] == "pass"
                        for sample in evaluations
                    ),
                    "dataset_hash": payload.get("hash", ""),
                    "error": "",
                }
    return functional


def lookup_summary(
    summary: list[dict[str, Any]],
    dataset: str,
    representation: str,
    tokenizer: str = "cl100k_base",
) -> dict[str, Any]:
    return next(
        row
        for row in summary
        if row["dataset"] == dataset
        and row["representation"] == representation
        and row["tokenizer"] == tokenizer
    )


def write_graphs(
    summary: list[dict[str, Any]],
    functional: dict[str, dict[str, dict[str, Any]]],
    representations: list[str],
    output_dir: Path,
) -> None:
    preferred_groups = ["HumanEval+", "MBPP+", "BigCodeBench"]
    available = {row["dataset"] for row in summary}
    groups = [group for group in preferred_groups if group in available]
    colors = {
        "python": "#64748b",
        "kern_compact": "#22c55e",
        "python_minifier": "#06b6d4",
        "sigil": "#f97316",
    }
    labels = {
        "python": "Python reference",
        "kern_compact": "Kern compact",
        "python_minifier": "python-minifier",
        "sigil": "Sigil 0.1.0",
    }
    comparison_representations = [
        representation
        for representation in representations
        if representation != "python"
    ]
    write_grouped_bar_svg(
        output_dir / "market-token-efficiency.svg",
        title="Token reduction under shared production tokenizers",
        subtitle=(
            "Same code-only Python corpus and cl100k_base; "
            "semantic coverage is reported separately"
        ),
        groups=groups,
        series=[
            (
                labels[representation],
                colors[representation],
                [
                    lookup_summary(
                        summary,
                        group,
                        representation,
                    )["saved_pct"]
                    for group in groups
                ],
            )
            for representation in comparison_representations
        ],
        y_label="Encoded tokens saved (%)",
        max_value=40.0,
    )
    write_grouped_bar_svg(
        output_dir / "market-structural-coverage.svg",
        title="Decoded-Python structural coverage",
        subtitle=(
            "Share of the full corpus that decodes to parseable Python; "
            "higher is better"
        ),
        groups=groups,
        series=[
            (
                labels[representation],
                colors[representation],
                [
                    (
                        lookup_summary(
                            summary,
                            group,
                            representation,
                        )["parse_ok"]
                        / lookup_summary(
                            summary,
                            group,
                            representation,
                        )["total_cases"]
                        * 100
                    )
                    for group in groups
                ],
            )
            for representation in comparison_representations
        ],
        y_label="Parseable round-trips (%)",
    )
    if not functional:
        return
    eval_groups = ["HumanEval+", "MBPP+"]
    write_grouped_bar_svg(
        output_dir / "market-evalplus-correctness.svg",
        title="Market comparison: EvalPlus preservation",
        subtitle=(
            "Official base + extra tests after each representation "
            "round-trip; full denominator"
        ),
        groups=eval_groups,
        series=[
            (
                labels[representation],
                colors[representation],
                [
                    (
                        functional[group][representation]["plus_pass"]
                        / functional[group][representation]["total"]
                        * 100
                    )
                    for group in eval_groups
                ],
            )
            for representation in representations
        ],
        y_label="Base + extra tests passed (%)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kern shared-corpus market benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/market"),
    )
    parser.add_argument("--skip-bigcodebench", action="store_true")
    parser.add_argument("--run-functional", action="store_true")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--evalplus-min-time-limit", type=float, default=5.0)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=DEFAULT_REPRESENTATIONS,
        default=list(DEFAULT_REPRESENTATIONS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    representations = list(dict.fromkeys(args.representations))
    adapters = build_adapters(representations)
    tasks = load_evalplus_tasks()
    if not args.skip_bigcodebench:
        tasks.extend(load_bigcodebench_tasks())
    encodings = {
        tokenizer: tiktoken.get_encoding(tokenizer)
        for tokenizer in TOKENIZERS
    }

    results: list[MarketCaseResult] = []
    for index, task in enumerate(tasks, start=1):
        for adapter in adapters:
            results.append(evaluate_case(task, adapter, encodings))
        if index % 100 == 0 or index == len(tasks):
            print(f"Market benchmark: {index}/{len(tasks)}")

    summary = aggregate(results, representations)
    functional: dict[str, dict[str, dict[str, Any]]] = {}
    if args.run_functional:
        functional = run_evalplus(
            tasks,
            results,
            representations,
            args.parallel,
            args.evalplus_min_time_limit,
        )

    failures = [result for result in results if result.error_stage]
    failure_keys = sorted(
        {
            (result.dataset, result.representation, result.error_stage)
            for result in failures
        }
    )
    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "same code-only source, same production tokenizers, "
                "full denominator, decoded-Python structural checks, "
                "optional official EvalPlus execution"
            ),
            "representations": {
                adapter.name: adapter.version for adapter in adapters
            },
            "tokenizers": list(TOKENIZERS),
            "evalplus": package_version("evalplus"),
            "humaneval_plus_hash": get_human_eval_plus_hash(),
            "mbpp_plus_hash": get_mbpp_plus_hash(),
            "functional_executed": bool(functional),
            "evalplus_min_time_limit": (
                args.evalplus_min_time_limit if functional else None
            ),
            "bigcodebench_split": (
                None if args.skip_bigcodebench else "v0.1.4"
            ),
            "bigcodebench_revision": (
                None
                if args.skip_bigcodebench
                else BIGCODEBENCH_REVISION
            ),
            "sigil_distribution": (
                "https://pypi.org/project/sigil-lang/"
                if "sigil" in representations
                else None
            ),
            "sigil_wheel_sha256": (
                SIGIL_WHEEL_SHA256
                if "sigil" in representations
                else None
            ),
            "normalization": (
                "code-only source after removing no-op string "
                "expressions/docstrings; remaining formatting preserved"
            ),
        },
        "summary": summary,
        "functional": functional,
        "failure_counts": [
            {
                "dataset": dataset,
                "representation": representation,
                "stage": stage,
                "count": sum(
                    result.dataset == dataset
                    and result.representation == representation
                    and result.error_stage == stage
                    for result in failures
                ),
            }
            for dataset, representation, stage in failure_keys
        ],
        "failure_examples": [
            {
                "dataset": result.dataset,
                "task_id": result.task_id,
                "representation": result.representation,
                "stage": result.error_stage,
                "message": result.error_message,
            }
            for result in failures[:50]
        ],
    }
    summary_path = args.output_dir / "market-benchmark-summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_detail_csv(
        results,
        args.output_dir / "market-benchmark-details.csv",
    )
    write_graphs(
        summary,
        functional,
        representations,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    if functional:
        print(json.dumps(functional, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
