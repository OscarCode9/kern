"""Reproducible public-example benchmark for Kern and NERD.

NERD's public table reports 49/32 NERD tokens versus 73/47 Python tokens on
FizzBuzz and four math functions.  The repository does not identify an LLM
tokenizer or publish the paired Python sources.  Its ``nerd tokens`` command
prints compiler lexer tokens.

This harness audits those counts and compares all seven deterministic local
NERD examples against matched Python, Kern compact, and python-minifier source
under shared production tokenizers.  Every program is compiled and executed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as svg_escape

import python_minifier
import tiktoken
from tokenizers import Tokenizer

from benchmark_modern import normalize_ast, write_grouped_bar_svg
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile
from train_kern_tokenizer import sha256_file, sha256_text

NERD_COMMIT = "edeafd53c4282a322bfe882bab05e7890e4766fd"
NERD_VERSION = "3.0.0"
EXPECTED_PAIRS = 7
LEXER_TOKEN_PATTERN = re.compile(r"(?:^| )([A-Z][A-Z0-9_]*)\(")


@dataclass(frozen=True)
class Pair:
    task_id: str
    nerd_example: str
    python: str
    nerd: str
    expected_stdout: str


@dataclass
class PairResult:
    task_id: str
    nerd_example: str
    python_sha256: str
    nerd_sha256: str
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    nerd_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    nerd_o200k: int
    kern_native_16k: int
    nerd_lexer_tokens: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    nerd_parse_ok: bool
    nerd_compile_run_ok: bool
    nerd_oracle_ok: bool
    nerd_error_stage: str
    nerd_error_message: str


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def strip_nerd_comments(value: str) -> str:
    return (
        "\n".join(
            line
            for line in value.splitlines()
            if not (
                line.lstrip().startswith("--")
                or line.lstrip().startswith("#")
            )
        ).strip()
        + "\n"
    )


def paired_programs() -> list[Pair]:
    pairs = [
        Pair(
            "public/calculator",
            "calculator.nerd",
            source(
                """
                def calc(a, b, op):
                    if op == 0:
                        return a + b
                    if op == 1:
                        return a - b
                    if op == 2:
                        return a * b
                    if op == 3:
                        return a / b
                    return 0

                print(calc(10, 5, 0))
                print(calc(10, 5, 1))
                print(calc(10, 5, 2))
                print(calc(10, 5, 3))
                """
            ),
            source(
                """
                fn calc a b op
                if op eq zero ret a plus b else if op eq one ret a minus b else if op eq two ret a times b else if op eq three ret a over b else ret zero

                fn main
                out call calc ten five zero
                out call calc ten five one
                out call calc ten five two
                out call calc ten five three
                """
            ),
            "15\n5\n50\n2",
        ),
        Pair(
            "public/conditionals",
            "conditionals.nerd",
            source(
                """
                def sign(x):
                    if x > 0:
                        return 1
                    if x < 0:
                        return -1
                    return 0

                def absolute(x):
                    return -x if x < 0 else x

                def maximum(a, b):
                    return a if a > b else b

                print(sign(5))
                print(sign(-3))
                print(sign(0))
                print(absolute(-7))
                print(maximum(3, 8))
                """
            ),
            source(
                """
                fn sign x
                if x gt zero ret one else if x lt zero ret neg one else ret zero

                fn abs x
                if x lt zero ret neg x else ret x

                fn max a b
                if a gt b ret a else ret b

                fn main
                out call sign five
                out call sign neg three
                out call sign zero
                out call abs neg seven
                out call max three eight
                """
            ),
            "1\n-1\n0\n7\n8",
        ),
        Pair(
            "public/fizzbuzz",
            "fizzbuzz.nerd",
            source(
                """
                def fizzbuzz(n):
                    for i in range(1, n + 1):
                        if i % 15 == 0:
                            print("FizzBuzz")
                        elif i % 3 == 0:
                            print("Fizz")
                        elif i % 5 == 0:
                            print("Buzz")
                        else:
                            print(i)

                fizzbuzz(15)
                """
            ),
            source(
                """
                fn fizzbuzz n
                repeat n times as i
                  if i mod 15 eq zero out "FizzBuzz" else if i mod three eq zero out "Fizz" else if i mod five eq zero out "Buzz" else out i
                done

                fn main
                call fizzbuzz 15
                """
            ),
            (
                "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n11\n"
                "Fizz\n13\n14\nFizzBuzz"
            ),
        ),
        Pair(
            "public/functions",
            "functions.nerd",
            source(
                """
                import math

                def square(x):
                    return x * x

                def cube(x):
                    squared = square(x)
                    return x * squared

                def double(x):
                    return x + x

                def hypotenuse(a, b):
                    a_squared = square(a)
                    b_squared = square(b)
                    total = a_squared + b_squared
                    return math.sqrt(total)

                print(square(4))
                print(cube(3))
                print(double(7))
                print(hypotenuse(3, 4))
                """
            ),
            source(
                """
                fn square x
                ret x times x

                fn cube x
                let sq call square x
                ret x times sq

                fn double x
                ret x plus x

                fn hypotenuse a b
                let asq call square a
                let bsq call square b
                let sum asq plus bsq
                ret math sqrt sum

                fn main
                out call square four
                out call cube three
                out call double seven
                out call hypotenuse three four
                """
            ),
            "16\n27\n14\n5",
        ),
        Pair(
            "public/loops",
            "loops.nerd",
            source(
                """
                def countdown(n):
                    for i in range(1, n + 1):
                        print(n - i + 1)

                def sumto(n):
                    total = 0
                    for i in range(1, n + 1):
                        total += i
                    return total

                def factorial(n):
                    result = 1
                    x = n
                    while x > 1:
                        result *= x
                        x -= 1
                    return result

                countdown(5)
                print(sumto(10))
                print(factorial(5))
                """
            ),
            source(
                """
                fn countdown n
                repeat n times as i
                  out n minus i plus one
                done

                fn sumto n
                let total zero
                repeat n times as i
                  inc total i
                done
                ret total

                fn factorial n
                let result one
                let x n
                while x gt one
                  let result result times x
                  dec x
                done
                ret result

                fn main
                call countdown five
                out call sumto ten
                out call factorial five
                """
            ),
            "5\n4\n3\n2\n1\n55\n120",
        ),
        Pair(
            "public/math",
            "math.nerd",
            source(
                """
                def add(a, b):
                    return a + b

                def sub(a, b):
                    return a - b

                def mul(a, b):
                    return a * b

                def div(a, b):
                    return a / b

                print(add(5, 3))
                print(sub(10, 4))
                print(mul(6, 7))
                print(div(15, 3))
                """
            ),
            source(
                """
                fn add a b
                ret a plus b

                fn sub a b
                ret a minus b

                fn mul a b
                ret a times b

                fn div a b
                ret a over b

                fn main
                out call add five three
                out call sub ten four
                out call mul six seven
                out call div 15 three
                """
            ),
            "8\n6\n42\n5",
        ),
        Pair(
            "public/output",
            "output.nerd",
            source(
                """
                print(42)
                print("hello world")
                print(5 + 3)
                """
            ),
            source(
                """
                out 42
                out "hello world"
                out five plus three
                """
            ),
            "42\nhello world\n8",
        ),
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} NERD pairs.")
    if len({pair.task_id for pair in pairs}) != EXPECTED_PAIRS:
        raise RuntimeError("NERD pair IDs must be unique.")
    return pairs


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_command(command: list[str]) -> tuple[bool, str, str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (
        result.returncode == 0,
        result.stdout.strip(),
        result.stderr.strip(),
    )


def normalize_stdout(value: str) -> str:
    """Normalize only whole-line numeric formatting such as 5.0 versus 5."""
    normalized: list[str] = []
    for line in value.strip().splitlines():
        stripped = line.strip()
        if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", stripped):
            try:
                number = Decimal(stripped)
                if number == number.to_integral_value():
                    normalized.append(str(number.quantize(Decimal(1))))
                else:
                    normalized.append(format(number.normalize(), "f"))
                continue
            except InvalidOperation:
                pass
        normalized.append(stripped)
    return "\n".join(normalized)


def lexer_token_count(nerd_binary: Path, source_path: Path) -> int:
    ok, stdout, stderr = run_command(
        [str(nerd_binary), "tokens", str(source_path)]
    )
    if not ok:
        raise RuntimeError(f"NERD lexer failed: {stderr}")
    names = LEXER_TOKEN_PATTERN.findall(stdout.replace("\n", " "))
    return sum(name != "EOF" for name in names)


def verify_public_sources(pairs: list[Pair], nerd_root: Path) -> None:
    for pair in pairs:
        path = nerd_root / "examples" / pair.nerd_example
        actual = strip_nerd_comments(path.read_text(encoding="utf-8"))
        if actual != pair.nerd:
            raise RuntimeError(
                f"Pinned public NERD example drifted: {pair.nerd_example}"
            )


def score_pairs(
    *,
    pairs: list[Pair],
    nerd_root: Path,
    tokenizer: Tokenizer,
) -> list[PairResult]:
    nerd_binary = nerd_root / "bootstrap" / "nerd"
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    results: list[PairResult] = []
    with tempfile.TemporaryDirectory(prefix="kern-nerd-pairs-") as directory:
        temp = Path(directory)
        for index, pair in enumerate(pairs):
            python_source = pair.python.strip()
            nerd_source = pair.nerd.strip()
            compact_tree_value = compact_tree(ast.parse(python_source))
            expected_compact = ast.unparse(compact_tree_value)
            kern_source = transpile(python_source, compact=True).strip()
            decoded = compile_kern(kern_source)
            minified = python_minifier.minify(
                python_source,
                rename_globals=False,
            )
            native_ids = tokenizer.encode(kern_source).ids

            python_path = temp / f"{index:02d}-python.py"
            kern_path = temp / f"{index:02d}-kern.py"
            minified_path = temp / f"{index:02d}-minified.py"
            nerd_path = temp / f"{index:02d}.nerd"
            python_path.write_text(python_source + "\n", encoding="utf-8")
            kern_path.write_text(decoded + "\n", encoding="utf-8")
            minified_path.write_text(minified + "\n", encoding="utf-8")
            nerd_path.write_text(nerd_source + "\n", encoding="utf-8")

            python_ok, python_stdout, _ = run_command(
                [sys.executable, str(python_path)]
            )
            kern_ok, kern_stdout, _ = run_command(
                [sys.executable, str(kern_path)]
            )
            minified_ok, minified_stdout, _ = run_command(
                [sys.executable, str(minified_path)]
            )
            parse_ok, _, parse_error = run_command(
                [str(nerd_binary), "parse", str(nerd_path)]
            )
            nerd_ok, nerd_stdout, nerd_error = run_command(
                [str(nerd_binary), "run", str(nerd_path)]
            )

            expected = normalize_stdout(pair.expected_stdout)
            error_stage = ""
            error_message = ""
            if not parse_ok:
                error_stage = "parse"
                error_message = parse_error
            elif not nerd_ok:
                error_stage = "compile_run"
                error_message = nerd_error
            elif normalize_stdout(nerd_stdout) != expected:
                error_stage = "oracle"
                error_message = (
                    f"expected {expected!r}, "
                    f"got {normalize_stdout(nerd_stdout)!r}"
                )

            results.append(
                PairResult(
                    task_id=pair.task_id,
                    nerd_example=pair.nerd_example,
                    python_sha256=sha256_text(python_source),
                    nerd_sha256=sha256_text(nerd_source),
                    python_cl100k=len(encodings["cl100k_base"].encode(python_source)),
                    kern_cl100k=len(encodings["cl100k_base"].encode(kern_source)),
                    python_minifier_cl100k=len(
                        encodings["cl100k_base"].encode(minified)
                    ),
                    nerd_cl100k=len(encodings["cl100k_base"].encode(nerd_source)),
                    python_o200k=len(encodings["o200k_base"].encode(python_source)),
                    kern_o200k=len(encodings["o200k_base"].encode(kern_source)),
                    python_minifier_o200k=len(
                        encodings["o200k_base"].encode(minified)
                    ),
                    nerd_o200k=len(encodings["o200k_base"].encode(nerd_source)),
                    kern_native_16k=len(native_ids),
                    nerd_lexer_tokens=lexer_token_count(
                        nerd_binary, nerd_path
                    ),
                    kern_native_exact_roundtrip=(
                        tokenizer.decode(native_ids) == kern_source
                    ),
                    kern_contract_ast=(
                        normalize_ast(decoded)
                        == normalize_ast(expected_compact)
                    ),
                    python_oracle_ok=(
                        python_ok
                        and normalize_stdout(python_stdout) == expected
                    ),
                    kern_oracle_ok=(
                        kern_ok
                        and normalize_stdout(kern_stdout) == expected
                    ),
                    python_minifier_oracle_ok=(
                        minified_ok
                        and normalize_stdout(minified_stdout) == expected
                    ),
                    nerd_parse_ok=parse_ok,
                    nerd_compile_run_ok=nerd_ok,
                    nerd_oracle_ok=(
                        nerd_ok and normalize_stdout(nerd_stdout) == expected
                    ),
                    nerd_error_stage=error_stage,
                    nerd_error_message=error_message[-1_000:],
                )
            )
    return results


def pct_below(value: int, baseline: int) -> float:
    return (baseline - value) / baseline * 100 if baseline else 0.0


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    cl = {
        "python": sum(result.python_cl100k for result in results),
        "kern_compact": sum(result.kern_cl100k for result in results),
        "python_minifier": sum(
            result.python_minifier_cl100k for result in results
        ),
        "nerd": sum(result.nerd_cl100k for result in results),
    }
    o = {
        "python": sum(result.python_o200k for result in results),
        "kern_compact": sum(result.kern_o200k for result in results),
        "python_minifier": sum(
            result.python_minifier_o200k for result in results
        ),
        "nerd": sum(result.nerd_o200k for result in results),
    }
    native_kern = sum(result.kern_native_16k for result in results)
    return {
        "programs": len(results),
        "cl100k_base": cl,
        "o200k_base": o,
        "native_system": {
            "kern_native_16k": native_kern,
            "nerd_cl100k_base": cl["nerd"],
        },
        "functional": {
            "python": sum(result.python_oracle_ok for result in results),
            "kern_compact": sum(result.kern_oracle_ok for result in results),
            "python_minifier": sum(
                result.python_minifier_oracle_ok for result in results
            ),
            "nerd_parse": sum(result.nerd_parse_ok for result in results),
            "nerd_compile_run": sum(
                result.nerd_compile_run_ok for result in results
            ),
            "nerd_oracle": sum(result.nerd_oracle_ok for result in results),
        },
        "structural": {
            "kern_contract_ast": sum(
                result.kern_contract_ast for result in results
            ),
            "kern_native_exact_roundtrip": sum(
                result.kern_native_exact_roundtrip for result in results
            ),
        },
        "comparisons": {
            "kern_cl100k_below_nerd_pct": pct_below(
                cl["kern_compact"], cl["nerd"]
            ),
            "kern_native_below_nerd_cl100k_pct": pct_below(
                native_kern, cl["nerd"]
            ),
            "nerd_cl100k_below_python_pct": pct_below(
                cl["nerd"], cl["python"]
            ),
            "kern_shared_wins": sum(
                result.kern_cl100k < result.nerd_cl100k
                for result in results
            ),
            "kern_shared_ties": sum(
                result.kern_cl100k == result.nerd_cl100k
                for result in results
            ),
            "kern_native_system_wins": sum(
                result.kern_native_16k < result.nerd_cl100k
                for result in results
            ),
            "median_per_pair_kern_cl100k_below_nerd_pct": statistics.median(
                pct_below(result.kern_cl100k, result.nerd_cl100k)
                for result in results
            ),
        },
    }


def claim_audit(
    *,
    pairs: list[Pair],
    nerd_root: Path,
    nerd_binary: Path,
) -> dict[str, Any]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    by_id = {pair.task_id: pair for pair in pairs}
    math_full = by_id["public/math"].nerd
    math_claim_source = math_full.split("\nfn main", maxsplit=1)[0].strip()
    fizz_source = by_id["public/fizzbuzz"].nerd.strip()
    with tempfile.TemporaryDirectory(prefix="nerd-claim-audit-") as directory:
        temp = Path(directory)
        math_path = temp / "math-claim.nerd"
        fizz_path = temp / "fizzbuzz.nerd"
        math_path.write_text(math_claim_source + "\n", encoding="utf-8")
        fizz_path.write_text(fizz_source + "\n", encoding="utf-8")
        math_lexer = lexer_token_count(nerd_binary, math_path)
        fizz_lexer = lexer_token_count(nerd_binary, fizz_path)

    searchable: dict[str, str] = {}
    for path in nerd_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            searchable[str(path.relative_to(nerd_root))] = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue
    tokenizer_hits = sorted(
        path
        for path, value in searchable.items()
        if "cl100k" in value.lower() or "tiktoken" in value.lower()
    )
    return {
        "published_table": {
            "fizzbuzz": {"nerd": 49, "python": 73},
            "math_four_functions": {"nerd": 32, "python": 47},
            "tokenizer_named": bool(tokenizer_hits),
            "paired_python_sources_published": False,
            "tokenizer_reference_hits": tokenizer_hits,
        },
        "current_sources": {
            "fizzbuzz": {
                "cl100k_base": len(
                    encodings["cl100k_base"].encode(fizz_source)
                ),
                "o200k_base": len(
                    encodings["o200k_base"].encode(fizz_source)
                ),
                "compiler_lexer_tokens": fizz_lexer,
                "published_nerd_tokens": 49,
                "published_count_reproduced": fizz_lexer == 49,
            },
            "math_four_functions": {
                "cl100k_base": len(
                    encodings["cl100k_base"].encode(math_claim_source)
                ),
                "o200k_base": len(
                    encodings["o200k_base"].encode(math_claim_source)
                ),
                "compiler_lexer_tokens": math_lexer,
                "published_nerd_tokens": 32,
                "published_count_reproduced": math_lexer == 32,
            },
        },
        "counter_semantics": (
            "The public `nerd tokens` command prints the compiler lexer token "
            "stream, including EOF; it does not invoke an LLM tokenizer"
        ),
    }


def official_gates(nerd_root: Path, nerd_binary: Path) -> dict[str, Any]:
    version_ok, version_stdout, version_error = run_command(
        [str(nerd_binary), "--version"]
    )
    expected_version = f"nerd {NERD_VERSION}"
    test_ok, test_stdout, test_error = run_command(
        ["make", "-C", str(nerd_root / "bootstrap"), "test"]
    )
    completed_marker = "=== Tests Complete ===" in test_stdout
    return {
        "compiler_version": {
            "ok": version_ok and version_stdout == expected_version,
            "stdout": version_stdout,
            "expected": expected_version,
            "error": version_error,
        },
        "make_test": {
            "ok": test_ok and completed_marker,
            "scope": (
                "tokenize, parse, and compile examples/math.nerd; "
                "not a unit-test suite"
            ),
            "completed_marker": completed_marker,
            "error": test_error[-1_000:],
        },
    }


def write_details(results: list[PairResult], path: Path) -> None:
    fields = list(asdict(results[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_token_svg(path: Path, aggregate_row: dict[str, Any]) -> None:
    values = aggregate_row["cl100k_base"]
    bars = [
        ("Python", values["python"], "#94a3b8"),
        ("python-minifier", values["python_minifier"], "#06b6d4"),
        ("NERD", values["nerd"], "#f59e0b"),
        ("Kern compact", values["kern_compact"], "#22c55e"),
    ]
    width, height = 980, 540
    left, right, top, bottom = 92, 34, 100, 108
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(value for _, value, _ in bars) * 1.15
    group_width = plot_width / len(bars)
    bar_width = min(116, group_width * 0.62)
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        "<title id=\"title\">Kern versus NERD public examples</title>",
        (
            "<desc id=\"desc\">Aggregate cl100k_base tokens on all seven "
            "deterministic public NERD examples; lower is better</desc>"
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            'font-weight="700">Kern versus NERD: shared tokenizer</text>'
        ),
        (
            f'<text x="{left}" y="66" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            "All 7 deterministic public NERD examples; lower is better</text>"
        ),
    ]
    for tick_index in range(6):
        value = max_value * tick_index / 5
        y = top + plot_height - value / max_value * plot_height
        elements.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.1f}" '
                    f'x2="{left + plot_width}" y2="{y:.1f}" '
                    'stroke="#25314d" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 12}" y="{y + 5:.1f}" '
                    'fill="#94a3b8" text-anchor="end" '
                    'font-family="Inter,system-ui,sans-serif" font-size="12">'
                    f"{round(value):,}</text>"
                ),
            ]
        )
    for index, (label, value, color) in enumerate(bars):
        center = left + group_width * (index + 0.5)
        bar_height = value / max_value * plot_height
        x = center - bar_width / 2
        y = top + plot_height - bar_height
        elements.extend(
            [
                (
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                    f'height="{bar_height:.1f}" rx="7" fill="{color}"/>'
                ),
                (
                    f'<text x="{center:.1f}" y="{y - 10:.1f}" '
                    'fill="#f8fafc" text-anchor="middle" '
                    'font-family="Inter,system-ui,sans-serif" font-size="15" '
                    f'font-weight="700">{value:,}</text>'
                ),
                (
                    f'<text x="{center:.1f}" y="{top + plot_height + 28}" '
                    'fill="#cbd5e1" text-anchor="middle" '
                    'font-family="Inter,system-ui,sans-serif" font-size="13">'
                    f"{svg_escape(label)}</text>"
                ),
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_functional_svg(path: Path, aggregate_row: dict[str, Any]) -> None:
    total = aggregate_row["programs"]
    functional = aggregate_row["functional"]
    write_grouped_bar_svg(
        path,
        title="Public-example functional execution",
        subtitle=(
            "All seven NERD programs compile to native code and match the "
            "same normalized stdout oracle"
        ),
        groups=["7 public examples"],
        series=[
            (
                "Kern compact",
                "#22c55e",
                [functional["kern_compact"] / total * 100],
            ),
            (
                "NERD native",
                "#f59e0b",
                [functional["nerd_oracle"] / total * 100],
            ),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nerd-root",
        type=Path,
        required=True,
        help="Pinned NERD source checkout with bootstrap/nerd built.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "benchmark_results/native-tokenizer/kern-16k-tokenizer.json"
        ),
    )
    parser.add_argument(
        "--tokenizer-manifest",
        type=Path,
        default=Path(
            "benchmark_results/native-tokenizer/"
            "kern-16k-training-manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/nerd"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual_commit = git_commit(args.nerd_root)
    if actual_commit != NERD_COMMIT:
        raise RuntimeError(
            f"NERD checkout drifted: {actual_commit} != {NERD_COMMIT}"
        )
    nerd_binary = args.nerd_root / "bootstrap" / "nerd"
    if not nerd_binary.exists():
        raise RuntimeError("Build NERD first with `make -C bootstrap`.")
    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern tokenizer SHA-256 does not match manifest.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    pairs = paired_programs()
    verify_public_sources(pairs, args.nerd_root)
    results = score_pairs(
        pairs=pairs,
        nerd_root=args.nerd_root,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    gates = official_gates(args.nerd_root, nerd_binary)
    failed_gates = [name for name, gate in gates.items() if not gate["ok"]]
    if failed_gates:
        raise RuntimeError(
            "NERD official gates failed: " + ", ".join(failed_gates)
        )
    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "All seven deterministic local NERD examples paired with "
                "equivalent Python; full denominator; shared production "
                "tokenizers; parse, native compile/run, normalized stdout"
            ),
            "nerd_commit": actual_commit,
            "nerd_version": NERD_VERSION,
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "claim_audit": claim_audit(
            pairs=pairs,
            nerd_root=args.nerd_root,
            nerd_binary=nerd_binary,
        ),
        "official_gates": gates,
        "public_examples": aggregate_row,
        "failure_stages": {
            stage: sum(result.nerd_error_stage == stage for result in results)
            for stage in sorted(
                {
                    result.nerd_error_stage
                    for result in results
                    if result.nerd_error_stage
                }
            )
        },
    }
    (args.output_dir / "nerd-benchmark-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "nerd-pair-corpus.json").write_text(
        json.dumps(
            [asdict(pair) for pair in pairs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_details(
        results,
        args.output_dir / "nerd-benchmark-details.csv",
    )
    write_token_svg(
        args.output_dir / "nerd-token-density.svg",
        aggregate_row,
    )
    write_functional_svg(
        args.output_dir / "nerd-functional-preservation.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
