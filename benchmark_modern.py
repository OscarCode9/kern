"""
Reproducible modern benchmark for Kern.

Corpora:
- EvalPlus HumanEval+ v0.1.10
- EvalPlus MBPP+ v0.2.0
- BigCodeBench v0.1.4

Representations:
- Python source (reference)
- Kern v0.4 (reversible Python -> Kern -> Python)
- Kern v0.4 compact (private-local alpha-renaming, then Kern round-trip)
- python-minifier (source-to-source market baseline)

Every Python reference is code-only: no-op string expressions/docstrings are
removed while the remaining source formatting is preserved.  This prevents
benchmark prompts and documentation from inflating the reported grammar
savings.  EvalPlus functional execution is optional because it is
substantially slower than the structural/token pass.
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
from typing import Any, Iterable

import python_minifier
import tiktoken
from datasets import load_dataset
from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)

from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


REPRESENTATIONS = ("python", "kern", "kern_compact", "python_minifier")
TOKENIZERS = ("cl100k_base", "o200k_base")
BIGCODEBENCH_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"


@dataclass(frozen=True)
class Task:
    dataset: str
    task_id: str
    entry_point: str
    source: str


@dataclass
class CaseResult:
    dataset: str
    task_id: str
    representation: str
    transform_ok: bool
    parse_ok: bool
    ast_equal: bool | None
    python_cl100k: int
    representation_cl100k: int
    python_o200k: int
    representation_o200k: int
    encoded: str
    decoded: str
    error_stage: str
    error_message: str


class DocstringStripper(ast.NodeTransformer):
    """Remove no-op string expressions from every statement body."""

    @staticmethod
    def _strip(body: list[ast.stmt]) -> list[ast.stmt]:
        return [
            stmt
            for stmt in body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]

    def _visit_body(self, node: ast.AST) -> ast.AST:
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body = self._strip(node.body)
        self.generic_visit(node)
        return node

    visit_Module = _visit_body
    visit_FunctionDef = _visit_body
    visit_AsyncFunctionDef = _visit_body
    visit_ClassDef = _visit_body


def normalize_ast(source: str) -> str:
    tree = DocstringStripper().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def code_only_source(source: str) -> str:
    """Remove no-op strings without reformatting the remaining Python."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)

    spans: list[tuple[int, int, str]] = []
    parents = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for parent in ast.walk(tree):
        if not isinstance(parent, parents):
            continue
        string_statements = [
            stmt
            for stmt in parent.body
            if isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ]
        non_strings = [
            stmt for stmt in parent.body if stmt not in string_statements
        ]
        for stmt in string_statements:
            start_line = stmt.lineno - 1
            end_line = stmt.end_lineno - 1
            start = line_offsets[start_line] + stmt.col_offset
            end = line_offsets[end_line] + stmt.end_col_offset
            prefix = lines[start_line][: stmt.col_offset]
            suffix = lines[end_line][stmt.end_col_offset :].strip()
            needs_pass = not non_strings
            if not prefix.strip() and not suffix:
                start = line_offsets[start_line]
                end = (
                    line_offsets[end_line + 1]
                    if end_line + 1 < len(line_offsets)
                    else len(source)
                )
                replacement = f"{prefix}pass\n" if needs_pass else ""
            else:
                replacement = "pass" if needs_pass else ""
            spans.append((start, end, replacement))

    for start, end, replacement in sorted(spans, reverse=True):
        source = source[:start] + replacement + source[end:]
    ast.parse(source)
    return source


def load_evalplus_tasks() -> list[Task]:
    tasks: list[Task] = []
    for problem in get_human_eval_plus().values():
        tasks.append(
            Task(
                dataset="HumanEval+",
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                source=code_only_source(
                    problem["prompt"] + problem["canonical_solution"]
                ),
            )
        )
    for problem in get_mbpp_plus().values():
        tasks.append(
            Task(
                dataset="MBPP+",
                task_id=problem["task_id"],
                entry_point=problem["entry_point"],
                source=code_only_source(problem["canonical_solution"]),
            )
        )
    return tasks


def load_bigcodebench_tasks() -> list[Task]:
    dataset = load_dataset(
        "bigcode/bigcodebench",
        revision=BIGCODEBENCH_REVISION,
        split="v0.1.4",
    )
    return [
        Task(
            dataset="BigCodeBench",
            task_id=row["task_id"],
            entry_point=row["entry_point"],
            source=code_only_source(row["code_prompt"] + row["canonical_solution"]),
        )
        for row in dataset
    ]


def transform(source: str, representation: str) -> tuple[str, str]:
    if representation == "python":
        return source, source
    if representation == "kern":
        encoded = transpile(source)
        return encoded, compile_kern(encoded)
    if representation == "kern_compact":
        encoded = transpile(source, compact=True)
        return encoded, compile_kern(encoded)
    if representation == "python_minifier":
        encoded = python_minifier.minify(
            source,
            rename_globals=False,
        )
        return encoded, encoded
    raise ValueError(f"Unknown representation: {representation}")


def evaluate_case(
    task: Task,
    representation: str,
    encodings: dict[str, Any],
) -> CaseResult:
    source_counts = {
        name: len(encoding.encode(task.source))
        for name, encoding in encodings.items()
    }
    try:
        encoded, decoded = transform(task.source, representation)
    except Exception as exc:
        return CaseResult(
            dataset=task.dataset,
            task_id=task.task_id,
            representation=representation,
            transform_ok=False,
            parse_ok=False,
            ast_equal=False if representation in {"kern", "kern_compact"} else None,
            python_cl100k=source_counts["cl100k_base"],
            representation_cl100k=0,
            python_o200k=source_counts["o200k_base"],
            representation_o200k=0,
            encoded="",
            decoded="",
            error_stage="transform",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    encoded_counts = {
        name: len(encoding.encode(encoded))
        for name, encoding in encodings.items()
    }
    try:
        ast.parse(decoded)
    except Exception as exc:
        return CaseResult(
            dataset=task.dataset,
            task_id=task.task_id,
            representation=representation,
            transform_ok=True,
            parse_ok=False,
            ast_equal=False if representation in {"kern", "kern_compact"} else None,
            python_cl100k=source_counts["cl100k_base"],
            representation_cl100k=encoded_counts["cl100k_base"],
            python_o200k=source_counts["o200k_base"],
            representation_o200k=encoded_counts["o200k_base"],
            encoded=encoded,
            decoded=decoded,
            error_stage="parse",
            error_message=f"{type(exc).__name__}: {exc}",
        )

    ast_equal: bool | None = None
    if representation in {"python", "kern", "kern_compact"}:
        try:
            expected = task.source
            if representation == "kern_compact":
                expected = ast.unparse(compact_tree(ast.parse(task.source)))
            ast_equal = normalize_ast(expected) == normalize_ast(decoded)
        except Exception as exc:
            return CaseResult(
                dataset=task.dataset,
                task_id=task.task_id,
                representation=representation,
                transform_ok=True,
                parse_ok=True,
                ast_equal=False,
                python_cl100k=source_counts["cl100k_base"],
                representation_cl100k=encoded_counts["cl100k_base"],
                python_o200k=source_counts["o200k_base"],
                representation_o200k=encoded_counts["o200k_base"],
                encoded=encoded,
                decoded=decoded,
                error_stage="ast",
                error_message=f"{type(exc).__name__}: {exc}",
            )

    return CaseResult(
        dataset=task.dataset,
        task_id=task.task_id,
        representation=representation,
        transform_ok=True,
        parse_ok=True,
        ast_equal=ast_equal,
        python_cl100k=source_counts["cl100k_base"],
        representation_cl100k=encoded_counts["cl100k_base"],
        python_o200k=source_counts["o200k_base"],
        representation_o200k=encoded_counts["o200k_base"],
        encoded=encoded,
        decoded=decoded,
        error_stage="" if ast_equal is not False else "ast",
        error_message="" if ast_equal is not False else "normalized AST differs",
    )


def aggregate(results: list[CaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted({result.dataset for result in results})
    for dataset in datasets:
        for representation in REPRESENTATIONS:
            subset = [
                result
                for result in results
                if result.dataset == dataset
                and result.representation == representation
            ]
            successful = [result for result in subset if result.transform_ok]
            for tokenizer in TOKENIZERS:
                source_attr = (
                    "python_cl100k"
                    if tokenizer == "cl100k_base"
                    else "python_o200k"
                )
                repr_attr = (
                    "representation_cl100k"
                    if tokenizer == "cl100k_base"
                    else "representation_o200k"
                )
                python_tokens = sum(getattr(result, source_attr) for result in successful)
                representation_tokens = sum(
                    getattr(result, repr_attr) for result in successful
                )
                saved_tokens = python_tokens - representation_tokens
                saved_pct = (
                    saved_tokens / python_tokens * 100 if python_tokens else 0.0
                )
                ast_rows = [result for result in subset if result.ast_equal is not None]
                rows.append(
                    {
                        "dataset": dataset,
                        "representation": representation,
                        "tokenizer": tokenizer,
                        "total_cases": len(subset),
                        "transform_ok": len(successful),
                        "parse_ok": sum(result.parse_ok for result in subset),
                        "ast_equal": (
                            sum(result.ast_equal is True for result in ast_rows)
                            if ast_rows
                            else None
                        ),
                        "ast_cases": len(ast_rows),
                        "python_tokens": python_tokens,
                        "representation_tokens": representation_tokens,
                        "saved_tokens": saved_tokens,
                        "saved_pct": round(saved_pct, 4),
                    }
                )
    return rows


def write_detail_csv(results: list[CaseResult], path: Path) -> None:
    fields = [
        "dataset",
        "task_id",
        "representation",
        "transform_ok",
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row[field] for field in fields})


def write_evalplus_samples(
    tasks: Iterable[Task],
    results: list[CaseResult],
    representation: str,
    dataset: str,
    path: Path,
) -> None:
    decoded = {
        (result.task_id, result.representation): result.decoded
        for result in results
        if result.transform_ok and result.parse_ok
    }
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            if task.dataset != dataset:
                continue
            solution = decoded.get((task.task_id, representation))
            if solution is None:
                solution = "raise RuntimeError('representation conversion failed')"
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
    results: list[CaseResult],
    parallel: int,
    min_time_limit: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    functional: dict[str, dict[str, dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="kern-evalplus-") as temp_dir:
        temp = Path(temp_dir)
        for dataset, cli_dataset in (("HumanEval+", "humaneval"), ("MBPP+", "mbpp")):
            functional[dataset] = {}
            for representation in REPRESENTATIONS:
                samples = temp / f"{cli_dataset}-{representation}.jsonl"
                write_evalplus_samples(
                    tasks, results, representation, dataset, samples
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
                # EvalPlus' default RLIMIT_AS can be below macOS' current
                # virtual-memory limit and make every canonical task time out.
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
                if completed.returncode != 0 or not result_path.exists():
                    functional[dataset][representation] = {
                        "total": sum(task.dataset == dataset for task in tasks),
                        "base_pass": 0,
                        "plus_pass": 0,
                        "error": (completed.stderr or completed.stdout)[-2000:],
                    }
                    continue
                payload = json.loads(result_path.read_text(encoding="utf-8"))
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


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    y_label: str,
    max_value: float = 100.0,
) -> None:
    width, height = 980, 540
    left, right, top, bottom = 92, 34, 92, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(1, len(groups))
    bar_w = min(68, group_w / (len(series) + 1))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{svg_escape(title)}</title>',
        f'<desc id="desc">{svg_escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="38" fill="#f8fafc" font-family="Inter,system-ui,sans-serif" font-size="24" font-weight="700">{svg_escape(title)}</text>',
        f'<text x="{left}" y="65" fill="#94a3b8" font-family="Inter,system-ui,sans-serif" font-size="14">{svg_escape(subtitle)}</text>',
    ]
    tick_step = 20 if max_value >= 80 else 10
    for tick in range(0, int(max_value) + 1, tick_step):
        y = top + plot_h - tick / max_value * plot_h
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#25314d" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" fill="#94a3b8" text-anchor="end" font-family="Inter,system-ui,sans-serif" font-size="12">{tick}</text>'
        )
    elements.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" fill="#94a3b8" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2:.1f})" font-family="Inter,system-ui,sans-serif" font-size="13">{svg_escape(y_label)}</text>'
    )
    for group_index, group in enumerate(groups):
        center = left + group_w * (group_index + 0.5)
        start = center - bar_w * len(series) / 2
        elements.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 28}" fill="#cbd5e1" text-anchor="middle" font-family="Inter,system-ui,sans-serif" font-size="14">{svg_escape(group)}</text>'
        )
        for series_index, (_, color, values) in enumerate(series):
            value = values[group_index]
            bar_h = max(0.0, min(max_value, value)) / max_value * plot_h
            x = start + series_index * bar_w + 4
            y = top + plot_h - bar_h
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 8:.1f}" height="{bar_h:.1f}" rx="5" fill="{color}"/>'
            )
            elements.append(
                f'<text x="{x + (bar_w - 8) / 2:.1f}" y="{max(top + 14, y - 8):.1f}" fill="#f8fafc" text-anchor="middle" font-family="Inter,system-ui,sans-serif" font-size="12" font-weight="600">{value:.1f}%</text>'
            )
    legend_x = left
    legend_y = height - 24
    for label, color, _ in series:
        elements.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" rx="3" fill="{color}"/>',
                f'<text x="{legend_x + 22}" y="{legend_y}" fill="#cbd5e1" font-family="Inter,system-ui,sans-serif" font-size="13">{svg_escape(label)}</text>',
            ]
        )
        legend_x += 190
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


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
    output_dir: Path,
) -> None:
    groups = ["HumanEval+", "MBPP+", "BigCodeBench"]
    write_grouped_bar_svg(
        output_dir / "modern-token-efficiency.svg",
        title="Token reduction on modern Python benchmarks",
        subtitle="Aggregate cl100k_base tokens vs the same Python sources; higher is better",
        groups=groups,
        series=[
            (
                "Kern compact",
                "#22c55e",
                [
                    lookup_summary(summary, group, "kern_compact")["saved_pct"]
                    for group in groups
                ],
            ),
            (
                "Kern reversible",
                "#7c3aed",
                [
                    lookup_summary(summary, group, "kern")["saved_pct"]
                    for group in groups
                ],
            ),
            (
                "python-minifier 3.2.0",
                "#06b6d4",
                [
                    lookup_summary(summary, group, "python_minifier")[
                        "saved_pct"
                    ]
                    for group in groups
                ],
            ),
        ],
        y_label="Tokens saved (%)",
        max_value=40.0,
    )

    if not functional:
        return
    eval_groups = ["HumanEval+", "MBPP+"]
    write_grouped_bar_svg(
        output_dir / "modern-evalplus-correctness.svg",
        title="EvalPlus functional preservation",
        subtitle="Official base + extra tests on transformed canonical solutions; higher is better",
        groups=eval_groups,
        series=[
            (
                "Python reference",
                "#64748b",
                [
                    functional[group]["python"]["plus_pass"]
                    / functional[group]["python"]["total"]
                    * 100
                    for group in eval_groups
                ],
            ),
            (
                "Kern compact",
                "#22c55e",
                [
                    functional[group]["kern_compact"]["plus_pass"]
                    / functional[group]["kern_compact"]["total"]
                    * 100
                    for group in eval_groups
                ],
            ),
            (
                "Kern round-trip",
                "#7c3aed",
                [
                    functional[group]["kern"]["plus_pass"]
                    / functional[group]["kern"]["total"]
                    * 100
                    for group in eval_groups
                ],
            ),
            (
                "python-minifier",
                "#06b6d4",
                [
                    functional[group]["python_minifier"]["plus_pass"]
                    / functional[group]["python_minifier"]["total"]
                    * 100
                    for group in eval_groups
                ],
            ),
        ],
        y_label="Base + extra tests passed (%)",
    )


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/modern"),
    )
    parser.add_argument("--skip-bigcodebench", action="store_true")
    parser.add_argument("--run-functional", action="store_true")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--evalplus-min-time-limit", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_evalplus_tasks()
    if not args.skip_bigcodebench:
        tasks.extend(load_bigcodebench_tasks())

    encodings = {
        tokenizer: tiktoken.get_encoding(tokenizer)
        for tokenizer in TOKENIZERS
    }
    results: list[CaseResult] = []
    for index, task in enumerate(tasks, start=1):
        for representation in REPRESENTATIONS:
            results.append(evaluate_case(task, representation, encodings))
        if index % 100 == 0 or index == len(tasks):
            print(f"Structural/token benchmark: {index}/{len(tasks)}")

    summary = aggregate(results)
    functional: dict[str, dict[str, dict[str, Any]]] = {}
    if args.run_functional:
        functional = run_evalplus(
            tasks,
            results,
            args.parallel,
            args.evalplus_min_time_limit,
        )

    payload = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "kern_grammar": "v0.4",
            "kern_compact": (
                "opt-in local alpha-renaming; module names and function "
                "parameters preserved; AST checked against compact_tree"
            ),
            "evalplus": package_version("evalplus"),
            "humaneval_plus_hash": get_human_eval_plus_hash(),
            "mbpp_plus_hash": get_mbpp_plus_hash(),
            "bigcodebench_split": (
                None if args.skip_bigcodebench else "v0.1.4"
            ),
            "bigcodebench_revision": (
                None if args.skip_bigcodebench else BIGCODEBENCH_REVISION
            ),
            "python_minifier": package_version("python-minifier"),
            "tiktoken": package_version("tiktoken"),
            "evalplus_min_time_limit": (
                args.evalplus_min_time_limit if args.run_functional else None
            ),
            "normalization": (
                "code-only source after removing no-op string "
                "expressions/docstrings; remaining formatting preserved"
            ),
        },
        "summary": summary,
        "functional": functional,
        "failures": [
            {
                "dataset": result.dataset,
                "task_id": result.task_id,
                "representation": result.representation,
                "stage": result.error_stage,
                "message": result.error_message,
            }
            for result in results
            if result.error_stage
        ],
    }
    (args.output_dir / "modern-benchmark-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_detail_csv(results, args.output_dir / "modern-benchmark-details.csv")
    write_graphs(summary, functional, args.output_dir)

    print(json.dumps(payload["summary"], indent=2))
    if functional:
        print(json.dumps(functional, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
