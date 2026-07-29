"""Reproducible public-pair benchmark for Kern and Toke.

Toke cannot enter Kern's Python-to-representation market harness because it
does not publish a Python-to-Toke converter.  Its public evaluation repository
does, however, contain 60 Python reference functions with matching generated
Toke programs.  This harness turns those references into equivalent JSON-CLI
programs, keeps every public pair in the denominator, and reports:

* shared ``cl100k_base`` and ``o200k_base`` token totals;
* Toke's separately labelled native 16K-BPE total;
* Kern round-trip structure;
* current Toke compiler compatibility on all 1,000 public solutions; and
* an optional deterministic functional smoke probe on the 60 public pairs.

The smoke inputs below are Kern-authored public probes.  They are not Toke's
private held-out tests and must not be described as a reproduction of Toke's
private functional evaluation.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape as svg_escape

import python_minifier
import tiktoken

from benchmark_modern import TOKENIZERS, normalize_ast, write_grouped_bar_svg
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


TOKE_COMMIT = "a3adcebddbdf4629b5289a6f317ac6678c6061c8"
TOKE_EVAL_COMMIT = "851f6d8b2cfedea22833f3787ad96c19e072e952"
TOKE_TOKENIZER_VERSION = "0.1.0"
TOKE_TOKENIZER_WHEEL_SHA256 = (
    "c33eee7501da85966f969dc0007afd30c9cc2a7f2a6fb5f8"
    "b769aa7762210fcb"
)
EXPECTED_PUBLIC_PAIRS = 60
EXPECTED_TOKE_SOLUTIONS = 1000
REPRESENTATIONS = ("python", "kern_compact", "python_minifier", "toke")


# One deterministic, valid-domain smoke input for every published Python pair.
PROBE_INPUTS: dict[str, Any] = {
    "task-a-0001": [1, 2, -3],
    "task-a-0002": [2, -3, 4],
    "task-a-0003": [5, -2, 9],
    "task-a-0004": [5, -2, 9],
    "task-a-0005": [2, 3, 8],
    "task-a-0006": [17, 5],
    "task-a-0007": [3, 4],
    "task-a-0008": [48, 18],
    "task-a-0009": [12, 18],
    "task-a-0010": 17,
    "task-a-0011": 10,
    "task-a-0012": 5,
    "task-a-0013": -7,
    "task-a-0014": 7,
    "task-a-0015": 17,
    "task-a-0016": [12, 0, 10],
    "task-a-0017": [-2, 3],
    "task-a-0018": [-2, -1, 0, 4],
    "task-a-0019": [-2, -1, 0, 4],
    "task-a-0020": [-2, 3, 4],
    "task-a-0021": [-2, 0, 3, 4],
    "task-a-0022": [-2, 0, 3, 4],
    "task-a-0023": [5, -2, 9],
    "task-a-0024": [-2, -1, 4, 6],
    "task-a-0025": [7, -3],
    "task-a-0026": [17, 5],
    "task-a-0027": -4,
    "task-a-0028": -3,
    "task-a-0029": [7, -3],
    "task-a-0030": [7, -3],
    "task-a-0031": -4,
    "task-a-0032": -3,
    "task-a-0033": [12, 3],
    "task-a-0034": [7, -3],
    "task-a-0035": [7, -3],
    "task-a-0036": [1, 2, 3],
    "task-a-0037": [1, 2, -1],
    "task-a-0038": [[1, 2, 3], [4, 5, 6]],
    "task-a-0039": [0, 1, 0],
    "task-a-0040": [-8, 3, 7],
    "task-a-0041": [-3, 1, 6],
    "task-a-0042": [3, 1, 2],
    "task-a-0043": [3, 1, 2],
    "task-a-0044": [1, 2, 3],
    "task-a-0045": [-2, 0, 3, 4],
    "task-a-0046": [-2, 0, 3, 4],
    "task-a-0047": [-3, -2, 0, 1, 4],
    "task-a-0048": [-3, -2, 0, 1, 4],
    "task-a-0049": [-2, 0, 3],
    "task-a-0050": [-2, 0, 3],
    "task-a-0051": [-2, 0, 3],
    "task-a-0052": [[1, 2, 3], 5],
    "task-a-0053": [3, 1, 3, 2, 1],
    "task-a-0054": [[1, 2, 3, 4], 1],
    "task-a-0055": [[1, 2, 3, 4], 1],
    "task-a-0056": [[1, 2], [3], []],
    "task-a-0057": [[1, 2, 3, 4, 5], 2],
    "task-a-0058": [[1, 3, 2, 5, 4], 3],
    "task-a-0059": [[1, 3, 2, 5, 4], 3],
    "task-a-0060": [9, 8],
}


@dataclass
class PairSources:
    task_id: str
    python: str
    kern_compact: str
    python_minifier: str
    toke: str


@dataclass
class PairResult:
    task_id: str
    python_sha256: str
    toke_sha256: str
    python_cl100k: int
    kern_compact_cl100k: int
    python_minifier_cl100k: int
    toke_cl100k: int
    python_o200k: int
    kern_compact_o200k: int
    python_minifier_o200k: int
    toke_o200k: int
    toke_native_tokens: int
    kern_decode_ok: bool
    kern_parse_ok: bool
    kern_contract_ast: bool
    toke_check_ok: bool
    toke_error_code: str
    python_probe_ok: bool | None = None
    kern_probe_ok: bool | None = None
    python_minifier_probe_ok: bool | None = None
    toke_probe_ok: bool | None = None
    toke_probe_stage: str = ""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def count_toke_tokens(
    text: str,
    *,
    normalise_strings: bool = False,
) -> int:
    """Load the optional native tokenizer only when its lane is executed."""
    try:
        from toke_tokenizer import count_tokens
    except ImportError as exc:
        raise RuntimeError(
            "Native Toke lane requires toke-tokenizer=="
            f"{TOKE_TOKENIZER_VERSION}."
        ) from exc
    return count_tokens(
        text,
        normalise_strings=normalise_strings,
    )


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def task_id_for(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in node.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "task"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ):
            continue
        return decorator.args[0].value
    return None


def clean_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    cleaned = copy.deepcopy(node)
    cleaned.decorator_list = []
    if (
        cleaned.body
        and isinstance(cleaned.body[0], ast.Expr)
        and isinstance(cleaned.body[0].value, ast.Constant)
        and isinstance(cleaned.body[0].value.value, str)
    ):
        cleaned.body = cleaned.body[1:]
    return cleaned


def loaded_names(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    return {
        child.id
        for child in ast.walk(clean_function(node))
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def import_bindings(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {
            alias.asname or alias.name.split(".", maxsplit=1)[0]
            for alias in node.names
        }
    return {alias.asname or alias.name for alias in node.names}


def build_python_programs(solutions_file: Path) -> dict[str, str]:
    """Extract the 60 public functions into equivalent JSON-CLI programs."""
    module = ast.parse(solutions_file.read_text(encoding="utf-8"))
    definitions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    imports = [
        node
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    tasks = {
        task_id: node
        for node in definitions.values()
        if (task_id := task_id_for(node)) is not None
    }

    programs: dict[str, str] = {}
    for task_id, function in sorted(tasks.items()):
        helpers: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        pending = list(loaded_names(function))
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            helper = definitions.get(name)
            if helper is None or task_id_for(helper) is not None:
                continue
            helpers.append(helper)
            pending.extend(loaded_names(helper))

        needed = loaded_names(function)
        for helper in helpers:
            needed.update(loaded_names(helper))

        body: list[ast.stmt] = [ast.parse("import json, sys").body[0]]
        body.extend(
            copy.deepcopy(node)
            for node in imports
            if import_bindings(node).intersection(needed)
        )
        body.extend(
            clean_function(helper)
            for helper in sorted(helpers, key=lambda item: item.lineno)
        )
        body.append(clean_function(function))
        wrapper = (
            "print(json.dumps("
            f"{function.name}(json.loads(sys.argv[1])),"
            "separators=(',', ':')))"
        )
        body.extend(ast.parse(wrapper).body)
        program = ast.Module(body=body, type_ignores=[])
        programs[task_id] = ast.unparse(ast.fix_missing_locations(program)) + "\n"

    return programs


def load_pairs(toke_eval: Path) -> list[PairSources]:
    baselines = (
        toke_eval
        / "benchmark"
        / "baselines"
        / "python"
        / "solutions.py"
    )
    programs = build_python_programs(baselines)
    if len(programs) != EXPECTED_PUBLIC_PAIRS:
        raise RuntimeError(
            "Pinned Toke evaluation corpus should expose "
            f"{EXPECTED_PUBLIC_PAIRS} Python pairs; found {len(programs)}."
        )
    if set(programs) != set(PROBE_INPUTS):
        missing = sorted(set(programs).symmetric_difference(PROBE_INPUTS))
        raise RuntimeError(f"Probe input coverage drifted: {missing}")

    solution_dir = toke_eval / "benchmark" / "solutions"
    pairs: list[PairSources] = []
    for task_id, python_source in sorted(programs.items()):
        toke_path = solution_dir / f"{task_id}.toke"
        if not toke_path.exists():
            raise RuntimeError(f"Missing paired Toke solution: {toke_path}")
        pairs.append(
            PairSources(
                task_id=task_id,
                python=python_source,
                kern_compact=transpile(python_source, compact=True),
                python_minifier=python_minifier.minify(
                    python_source,
                    rename_globals=False,
                ),
                toke=toke_path.read_text(encoding="utf-8").strip(),
            )
        )
    return pairs


def first_error_code(output: str) -> str:
    for line in output.splitlines():
        if not line.lstrip().startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("severity") == "error":
            return str(record.get("error_code", "unknown"))
    return ""


def check_toke(
    compiler: Path,
    source_path: Path,
    timeout: float,
) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [str(compiler), "--legacy", "--check", str(source_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, first_error_code(output)


def evaluate_pairs(
    pairs: Iterable[PairSources],
    compiler: Path,
    toke_eval: Path,
    encodings: dict[str, Any],
    timeout: float,
) -> list[PairResult]:
    results: list[PairResult] = []
    solution_dir = toke_eval / "benchmark" / "solutions"
    for pair in pairs:
        decoded_ok = parse_ok = contract_ast = False
        try:
            decoded = compile_kern(pair.kern_compact)
            decoded_ok = True
            ast.parse(decoded)
            parse_ok = True
            expected = ast.unparse(compact_tree(ast.parse(pair.python)))
            contract_ast = normalize_ast(decoded) == normalize_ast(expected)
        except (SyntaxError, ValueError):
            pass

        check_ok, error_code = check_toke(
            compiler,
            solution_dir / f"{pair.task_id}.toke",
            timeout,
        )
        results.append(
            PairResult(
                task_id=pair.task_id,
                python_sha256=sha256_text(pair.python),
                toke_sha256=sha256_text(pair.toke),
                python_cl100k=len(
                    encodings["cl100k_base"].encode(pair.python)
                ),
                kern_compact_cl100k=len(
                    encodings["cl100k_base"].encode(pair.kern_compact)
                ),
                python_minifier_cl100k=len(
                    encodings["cl100k_base"].encode(pair.python_minifier)
                ),
                toke_cl100k=len(
                    encodings["cl100k_base"].encode(pair.toke)
                ),
                python_o200k=len(
                    encodings["o200k_base"].encode(pair.python)
                ),
                kern_compact_o200k=len(
                    encodings["o200k_base"].encode(pair.kern_compact)
                ),
                python_minifier_o200k=len(
                    encodings["o200k_base"].encode(pair.python_minifier)
                ),
                toke_o200k=len(
                    encodings["o200k_base"].encode(pair.toke)
                ),
                toke_native_tokens=count_toke_tokens(pair.toke),
                kern_decode_ok=decoded_ok,
                kern_parse_ok=parse_ok,
                kern_contract_ast=contract_ast,
                toke_check_ok=check_ok,
                toke_error_code=error_code,
            )
        )
    return results


def run_program(
    command: list[str],
    probe: Any,
    timeout: float,
) -> tuple[bool, Any, str]:
    try:
        result = subprocess.run(
            command + [json.dumps(probe, separators=(",", ":"))],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    if result.returncode != 0:
        return False, None, f"exit_{result.returncode}"
    output = result.stdout.strip()
    try:
        return True, json.loads(output), ""
    except json.JSONDecodeError:
        return False, None, "invalid_json_output"


def write_source(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def run_functional_probes(
    pairs: list[PairSources],
    results: list[PairResult],
    compiler: Path,
    timeout: float,
) -> None:
    by_id = {result.task_id: result for result in results}
    with tempfile.TemporaryDirectory(prefix="kern-toke-probes-") as raw_dir:
        root = Path(raw_dir)
        for pair in pairs:
            result = by_id[pair.task_id]
            probe = PROBE_INPUTS[pair.task_id]
            python_path = root / f"{pair.task_id}-python.py"
            kern_path = root / f"{pair.task_id}-kern.py"
            minifier_path = root / f"{pair.task_id}-minifier.py"
            write_source(python_path, pair.python)
            write_source(kern_path, compile_kern(pair.kern_compact))
            write_source(minifier_path, pair.python_minifier)

            oracle_ok, oracle, _ = run_program(
                [sys.executable, "-I", str(python_path)],
                probe,
                timeout,
            )
            result.python_probe_ok = oracle_ok
            for path, attr in (
                (kern_path, "kern_probe_ok"),
                (minifier_path, "python_minifier_probe_ok"),
            ):
                ok, value, _ = run_program(
                    [sys.executable, "-I", str(path)],
                    probe,
                    timeout,
                )
                setattr(result, attr, oracle_ok and ok and value == oracle)

            if not result.toke_check_ok:
                result.toke_probe_ok = False
                result.toke_probe_stage = "check"
                continue

            binary = root / f"{pair.task_id}-toke"
            toke_source = root / f"{pair.task_id}.toke"
            write_source(toke_source, pair.toke)
            try:
                compile_result = subprocess.run(
                    [
                        str(compiler),
                        "--legacy",
                        str(toke_source),
                        "--out",
                        str(binary),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout * 4,
                )
            except subprocess.TimeoutExpired:
                result.toke_probe_ok = False
                result.toke_probe_stage = "compile_timeout"
                continue
            if compile_result.returncode != 0 or not binary.exists():
                result.toke_probe_ok = False
                result.toke_probe_stage = "compile"
                continue

            ok, value, stage = run_program(
                [str(binary)],
                probe,
                timeout,
            )
            result.toke_probe_ok = oracle_ok and ok and value == oracle
            result.toke_probe_stage = (
                "pass"
                if result.toke_probe_ok
                else (stage or "output_mismatch")
            )


def aggregate_tokens(results: list[PairResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tokenizer, suffix in (
        ("cl100k_base", "cl100k"),
        ("o200k_base", "o200k"),
    ):
        python_tokens = sum(
            getattr(result, f"python_{suffix}") for result in results
        )
        for representation in REPRESENTATIONS:
            field = (
                f"{representation}_{suffix}"
                if representation != "python"
                else f"python_{suffix}"
            )
            representation_tokens = sum(
                getattr(result, field) for result in results
            )
            rows.append(
                {
                    "tokenizer": tokenizer,
                    "representation": representation,
                    "programs": len(results),
                    "python_tokens": python_tokens,
                    "representation_tokens": representation_tokens,
                    "saved_tokens": python_tokens - representation_tokens,
                    "saved_pct": (
                        (python_tokens - representation_tokens)
                        / python_tokens
                        * 100
                    ),
                }
            )
    return rows


def audit_public_toke_corpus(
    toke_eval: Path,
    compiler: Path,
    encodings: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    solution_dir = toke_eval / "benchmark" / "solutions"
    paths = sorted(solution_dir.glob("task-a-*.toke"))
    if len(paths) != EXPECTED_TOKE_SOLUTIONS:
        raise RuntimeError(
            "Pinned Toke evaluation corpus should expose "
            f"{EXPECTED_TOKE_SOLUTIONS} solutions; found {len(paths)}."
        )

    error_counts: Counter[str] = Counter()
    check_pass = 0
    cl100k_total = 0
    o200k_total = 0
    native_total = 0
    native_normalized_total = 0
    for path in paths:
        source = path.read_text(encoding="utf-8").strip()
        cl100k_total += len(encodings["cl100k_base"].encode(source))
        o200k_total += len(encodings["o200k_base"].encode(source))
        native_total += count_toke_tokens(source)
        native_normalized_total += count_toke_tokens(
            source,
            normalise_strings=True,
        )
        ok, error_code = check_toke(compiler, path, timeout)
        check_pass += int(ok)
        if not ok:
            error_counts[error_code or "unknown"] += 1

    official_csv = toke_eval / "data" / "gate1_token_counts.csv"
    with official_csv.open(newline="", encoding="utf-8") as handle:
        official_rows = list(csv.DictReader(handle))
    official_pass = sum(
        float(row["pass1"]) == 1.0 for row in official_rows
    )
    official_fail = sum(
        float(row["pass1"]) == 0.0 for row in official_rows
    )
    official_missing = sum(
        float(row["pass1"]) < 0.0 for row in official_rows
    )
    official_token_total = sum(
        int(row["token_count"]) for row in official_rows
    )

    return {
        "solutions": len(paths),
        "cl100k_base_tokens": cl100k_total,
        "o200k_base_tokens": o200k_total,
        "toke_native_tokens": native_total,
        "toke_native_normalized_tokens": native_normalized_total,
        "native_reduction_vs_cl100k_pct": (
            (cl100k_total - native_total) / cl100k_total * 100
        ),
        "current_compiler_check_pass": check_pass,
        "current_compiler_check_fail": len(paths) - check_pass,
        "current_compiler_first_error_counts": dict(
            error_counts.most_common()
        ),
        "official_gate1_csv": {
            "rows": len(official_rows),
            "cl100k_base_tokens": official_token_total,
            "pass1": official_pass,
            "fail": official_fail,
            "missing": official_missing,
        },
    }


def lookup_aggregate(
    rows: list[dict[str, Any]],
    tokenizer: str,
    representation: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["tokenizer"] == tokenizer
        and row["representation"] == representation
    )


def write_token_graph(
    path: Path,
    aggregates: list[dict[str, Any]],
) -> None:
    width, height = 1040, 600
    left, right, top = 190, 120, 92
    plot_width = width - left - right
    labels: list[tuple[str, str, str, int]] = []
    colors = {
        "python": "#64748b",
        "kern_compact": "#22c55e",
        "python_minifier": "#06b6d4",
        "toke": "#f97316",
    }
    display = {
        "python": "Python",
        "kern_compact": "Kern compact",
        "python_minifier": "python-minifier",
        "toke": "Toke",
    }
    for tokenizer in TOKENIZERS:
        for representation in REPRESENTATIONS:
            row = lookup_aggregate(
                aggregates,
                tokenizer,
                representation,
            )
            labels.append(
                (
                    tokenizer,
                    display[representation],
                    colors[representation],
                    row["representation_tokens"],
                )
            )
    max_value = max(value for _, _, _, value in labels)
    row_height = 52
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">Kern vs Toke on 60 public pairs</title>',
        (
            '<desc id="desc">Total tokens under the same production '
            'tokenizer; lower is better.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            'font-weight="700">Kern vs Toke on 60 public pairs</text>'
        ),
        (
            f'<text x="{left}" y="65" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            'Equivalent JSON-CLI programs; same tokenizer; lower is better'
            "</text>"
        ),
    ]
    for index, (tokenizer, label, color, value) in enumerate(labels):
        y = top + index * row_height
        bar_width = value / max_value * plot_width
        if index in (0, 4):
            elements.append(
                f'<text x="18" y="{y + 25}" fill="#94a3b8" '
                'font-family="Inter,system-ui,sans-serif" font-size="13">'
                f"{svg_escape(tokenizer)}</text>"
            )
        elements.extend(
            [
                (
                    f'<text x="{left - 14}" y="{y + 25}" fill="#cbd5e1" '
                    'text-anchor="end" font-family="Inter,system-ui,sans-serif" '
                    f'font-size="13">{svg_escape(label)}</text>'
                ),
                (
                    f'<rect x="{left}" y="{y + 7}" width="{bar_width:.1f}" '
                    f'height="28" rx="5" fill="{color}"/>'
                ),
                (
                    f'<text x="{left + bar_width + 12:.1f}" y="{y + 27}" '
                    'fill="#f8fafc" font-family="Inter,system-ui,sans-serif" '
                    f'font-size="13" font-weight="600">{value:,}</text>'
                ),
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_graphs(
    output_dir: Path,
    aggregates: list[dict[str, Any]],
    results: list[PairResult],
) -> None:
    write_token_graph(
        output_dir / "toke-shared-tokenizer.svg",
        aggregates,
    )
    if results[0].python_probe_ok is None:
        return
    total = len(results)
    passes = {
        "Python": sum(result.python_probe_ok is True for result in results),
        "Kern compact": sum(
            result.kern_probe_ok is True for result in results
        ),
        "python-minifier": sum(
            result.python_minifier_probe_ok is True for result in results
        ),
        "Toke": sum(result.toke_probe_ok is True for result in results),
    }
    write_grouped_bar_svg(
        output_dir / "toke-functional-probe.svg",
        title="Public-pair functional smoke probe",
        subtitle=(
            "One Kern-authored deterministic input per public pair; "
            "not Toke's private held-out suite"
        ),
        groups=["60 public pairs"],
        series=[
            ("Python", "#64748b", [passes["Python"] / total * 100]),
            (
                "Kern compact",
                "#22c55e",
                [passes["Kern compact"] / total * 100],
            ),
            (
                "python-minifier",
                "#06b6d4",
                [passes["python-minifier"] / total * 100],
            ),
            ("Toke", "#f97316", [passes["Toke"] / total * 100]),
        ],
        y_label="Programs matching Python (%)",
    )


def write_details(path: Path, results: list[PairResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def functional_summary(results: list[PairResult]) -> dict[str, Any]:
    if results[0].python_probe_ok is None:
        return {}
    return {
        representation: {
            "passed": sum(
                getattr(result, field) is True for result in results
            ),
            "total": len(results),
        }
        for representation, field in (
            ("python", "python_probe_ok"),
            ("kern_compact", "kern_probe_ok"),
            ("python_minifier", "python_minifier_probe_ok"),
            ("toke", "toke_probe_ok"),
        )
    }


def robustness_summary(results: list[PairResult]) -> dict[str, Any]:
    largest = max(results, key=lambda result: result.toke_cl100k)
    without_largest = [
        result for result in results if result.task_id != largest.task_id
    ]
    kern_without = sum(
        result.kern_compact_cl100k for result in without_largest
    )
    toke_without = sum(result.toke_cl100k for result in without_largest)
    ratios = [
        result.kern_compact_cl100k / result.toke_cl100k
        for result in results
    ]
    native_ratios = [
        result.kern_compact_cl100k / result.toke_native_tokens
        for result in results
    ]
    return {
        "kern_shared_tokenizer_wins": sum(
            result.kern_compact_cl100k < result.toke_cl100k
            for result in results
        ),
        "total_pairs": len(results),
        "median_per_pair_kern_below_toke_pct": (
            1 - statistics.median(ratios)
        )
        * 100,
        "largest_toke_source": {
            "task_id": largest.task_id,
            "toke_cl100k_base_tokens": largest.toke_cl100k,
            "kern_cl100k_base_tokens": largest.kern_compact_cl100k,
        },
        "excluding_largest_toke_source": {
            "programs": len(without_largest),
            "kern_cl100k_base_tokens": kern_without,
            "toke_cl100k_base_tokens": toke_without,
            "kern_below_toke_pct": (
                (toke_without - kern_without) / toke_without * 100
                if toke_without
                else None
            ),
        },
        "kern_cl100k_below_toke_native_pairs": sum(
            ratio < 1 for ratio in native_ratios
        ),
        "median_per_pair_kern_below_toke_native_pct": (
            1 - statistics.median(native_ratios)
        )
        * 100,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toke-eval", type=Path, required=True)
    parser.add_argument("--toke-compiler", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/toke"),
    )
    parser.add_argument("--run-functional", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Allow source repositories at commits other than the pinned ones.",
    )
    args = parser.parse_args(argv)

    toke_eval = args.toke_eval.resolve()
    compiler = args.toke_compiler.resolve()
    if not compiler.exists():
        parser.error(f"Toke compiler not found: {compiler}")
    toke_root = compiler.parent
    actual_toke_commit = git_commit(toke_root)
    actual_eval_commit = git_commit(toke_eval)
    if not args.allow_unpinned:
        if actual_toke_commit != TOKE_COMMIT:
            parser.error(
                f"Toke source must be at {TOKE_COMMIT}; "
                f"found {actual_toke_commit}."
            )
        if actual_eval_commit != TOKE_EVAL_COMMIT:
            parser.error(
                f"toke-eval source must be at {TOKE_EVAL_COMMIT}; "
                f"found {actual_eval_commit}."
            )
    tokenizer_version = package_version("toke-tokenizer")
    if tokenizer_version != TOKE_TOKENIZER_VERSION:
        parser.error(
            "Native lane requires "
            f"toke-tokenizer=={TOKE_TOKENIZER_VERSION}; "
            f"found {tokenizer_version}."
        )

    version_result = subprocess.run(
        [str(compiler), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    encodings = {
        tokenizer: tiktoken.get_encoding(tokenizer)
        for tokenizer in TOKENIZERS
    }
    pairs = load_pairs(toke_eval)
    results = evaluate_pairs(
        pairs,
        compiler,
        toke_eval,
        encodings,
        args.timeout,
    )
    if args.run_functional:
        run_functional_probes(
            pairs,
            results,
            compiler,
            args.timeout,
        )
    aggregates = aggregate_tokens(results)
    public_audit = audit_public_toke_corpus(
        toke_eval,
        compiler,
        encodings,
        args.timeout,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_details(
        output_dir / "toke-public-pair-details.csv",
        results,
    )
    write_graphs(output_dir, aggregates, results)

    kern_cl100k = lookup_aggregate(
        aggregates,
        "cl100k_base",
        "kern_compact",
    )["representation_tokens"]
    toke_cl100k = lookup_aggregate(
        aggregates,
        "cl100k_base",
        "toke",
    )["representation_tokens"]
    toke_native = sum(result.toke_native_tokens for result in results)
    summary = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "60 public Python/Toke pairs as equivalent JSON-CLI programs; "
                "full denominator; same production tokenizer in shared lane"
            ),
            "toke_commit": actual_toke_commit,
            "toke_eval_commit": actual_eval_commit,
            "toke_compiler_version": version_result.stdout.strip(),
            "toke_tokenizer_version": tokenizer_version,
            "toke_tokenizer_wheel_sha256": TOKE_TOKENIZER_WHEEL_SHA256,
            "python_version": sys.version.split()[0],
            "python_minifier_version": package_version("python-minifier"),
            "tiktoken_version": package_version("tiktoken"),
            "tokenizers": list(TOKENIZERS),
            "public_pairs": len(results),
            "functional_probe": args.run_functional,
            "functional_limit": (
                "Kern-authored smoke probes, not Toke private held-out tests"
            ),
        },
        "shared_tokenizer_aggregates": aggregates,
        "native_tokenizer_lane": {
            "scope": "Toke source only; separately labeled cross-tokenizer row",
            "programs": len(results),
            "toke_native_tokens": toke_native,
            "toke_cl100k_base_tokens": toke_cl100k,
            "native_reduction_vs_toke_cl100k_pct": (
                (toke_cl100k - toke_native) / toke_cl100k * 100
            ),
            "kern_cl100k_base_tokens": kern_cl100k,
            "kern_below_toke_native_pct": (
                (toke_native - kern_cl100k) / toke_native * 100
            ),
            "fairness_note": (
                "Kern and Toke use different tokenizers in this observation; "
                "the shared-tokenizer aggregates are the neutral language lane."
            ),
        },
        "robustness": robustness_summary(results),
        "structural": {
            "kern_decode_ok": sum(
                result.kern_decode_ok for result in results
            ),
            "kern_parse_ok": sum(
                result.kern_parse_ok for result in results
            ),
            "kern_contract_ast": sum(
                result.kern_contract_ast for result in results
            ),
            "toke_current_compiler_check_ok": sum(
                result.toke_check_ok for result in results
            ),
            "total": len(results),
            "toke_first_error_counts": dict(
                Counter(
                    result.toke_error_code
                    for result in results
                    if not result.toke_check_ok
                ).most_common()
            ),
        },
        "functional_probe": functional_summary(results),
        "toke_public_corpus_audit": public_audit,
    }
    (output_dir / "toke-public-pair-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "shared_tokenizer_aggregates": aggregates,
                "native_tokenizer_lane": summary["native_tokenizer_lane"],
                "robustness": summary["robustness"],
                "structural": summary["structural"],
                "functional_probe": summary["functional_probe"],
                "toke_public_corpus_audit": public_audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
