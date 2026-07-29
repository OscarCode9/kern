"""Reproducible paired benchmark for Kern and KARN v1.0.0.

KARN publishes a 76% token-reduction claim but not the paired REST API sources
or tokenizer configuration behind its approximate 47-vs-198 token example.
This harness therefore keeps two lanes separate:

1. a claim audit over the exact current public KARN snippets and files; and
2. a 46-program matched corpus derived from KARN's public examples and
   interpreter conformance tests, with deterministic stdout oracles.

Every representation remains in the denominator.  KARN's interpreter and
Python code-generation target are scored independently.
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

KARN_COMMIT = "5208a5f592083c4885281b1d505af06fc58995ba"
KARN_VERSION = "1.0.0"
EXPECTED_PAIRS = 46
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

PUBLIC_API_SNIPPET = """
@web  #http #db.pg #auth

type User:{id:N, name:S, role:S}

^getUser->req:
  tok  = auth.verify(req.header.token)?
  user = db.q("users", {id:req.p.id})?
  !user

http.serve(3000, {"/users/:id": getUser})
"""


@dataclass(frozen=True)
class Pair:
    task_id: str
    origin: str
    python: str
    karn: str
    expected_stdout: str


@dataclass
class PairResult:
    task_id: str
    origin: str
    python_sha256: str
    karn_sha256: str
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    karn_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    karn_o200k: int
    kern_native_16k: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    karn_check_ok: bool
    karn_interpreter_oracle_ok: bool
    karn_python_codegen_ok: bool
    karn_python_codegen_oracle_ok: bool
    karn_error_stage: str
    karn_error_message: str


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def paired_programs() -> list[Pair]:
    """Return fixed pairs derived from KARN examples and conformance tests."""
    pairs = [
        Pair(
            "public/hello",
            "KARN example hello.kn",
            source('print("Hello from KARN!")'),
            source('! "Hello from KARN!"'),
            "Hello from KARN!",
        ),
        Pair(
            "public/fibonacci",
            "KARN example fibonacci.kn",
            source(
                """
                def fib(n):
                    if n == 0:
                        return 0
                    if n == 1:
                        return 1
                    return fib(n - 1) + fib(n - 2)

                print(fib(10))
                """
            ),
            source(
                """
                fib->n:N:N
                  match n{
                    0 -> 0
                    1 -> 1
                    x -> fib(x - 1) + fib(x - 2)
                  }

                ! fib(10)
                """
            ),
            "55",
        ),
        Pair(
            "public/collections",
            "KARN example collections.kn",
            source(
                """
                items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                doubled = [x * 2 for x in items]
                evens = [x for x in items if x % 2 == 0]
                result = [x for x in [x * 3 for x in items] if x > 10]
                one_to_hundred = list(range(1, 101))
                tagged = ("Ok", 42)
                print(tagged[1] if tagged[0] == "Ok" else -1)
                """
            ),
            source(
                """
                items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                doubled = items*(x -> x * 2)
                evens = items%(x -> x % 2 == 0)
                result = items*(x -> x * 3)%(x -> x > 10)
                one_to_hundred = 1..100
                match Ok(42){
                  Ok(v)  -> !v
                  Err(e) -> !(-1)
                }
                """
            ),
            "42",
        ),
        Pair(
            "literal/number",
            "KARN interpreter conformance",
            source("print(42)"),
            source("! 42"),
            "42",
        ),
        Pair(
            "literal/string",
            "KARN interpreter conformance",
            source('print("hello")'),
            source('! "hello"'),
            "hello",
        ),
        Pair(
            "arithmetic/precedence",
            "KARN interpreter conformance",
            source("print(1 + 2 * 3)"),
            source("! 1 + 2 * 3"),
            "7",
        ),
        Pair(
            "binding/immutable",
            "KARN interpreter conformance",
            source("x = 42\nprint(x)"),
            source("x = 42\n! x"),
            "42",
        ),
        Pair(
            "binding/mutable",
            "KARN interpreter conformance",
            source("x = 1\nx = 2\nprint(x)"),
            source("~x = 1\n~x = 2\n! x"),
            "2",
        ),
        Pair(
            "function/add",
            "KARN interpreter conformance",
            source(
                """
                def add(a, b):
                    return a + b

                print(add(3, 4))
                """
            ),
            source("add->a b: a+b\n! add(3, 4)"),
            "7",
        ),
        Pair(
            "comparison/lt",
            "KARN interpreter conformance",
            source("print(3 < 5)"),
            source("! 3 < 5"),
            "True",
        ),
        Pair(
            "comparison/gt",
            "KARN interpreter conformance",
            source("print(7 > 5)"),
            source("! 7 > 5"),
            "True",
        ),
        Pair(
            "comparison/lte",
            "KARN interpreter conformance",
            source("print(5 <= 5)"),
            source("! 5 <= 5"),
            "True",
        ),
        Pair(
            "comparison/gte",
            "KARN interpreter conformance",
            source("print(7 >= 5)"),
            source("! 7 >= 5"),
            "True",
        ),
        Pair(
            "comparison/eq",
            "KARN interpreter conformance",
            source("print(42 == 42)"),
            source("! 42 == 42"),
            "True",
        ),
        Pair(
            "comparison/neq",
            "KARN interpreter conformance",
            source("print(1 != 2)"),
            source("! 1 != 2"),
            "True",
        ),
        Pair(
            "collection/range",
            "KARN interpreter conformance",
            source("print(list(range(1, 6)))"),
            source("! 1..5"),
            "[1, 2, 3, 4, 5]",
        ),
        Pair(
            "collection/list",
            "KARN interpreter conformance",
            source("print([1, 2, 3])"),
            source("! [1, 2, 3]"),
            "[1, 2, 3]",
        ),
        Pair(
            "collection/map",
            "KARN interpreter conformance",
            source('print({"x": 1, "y": 2})'),
            source("! {x:1, y:2}"),
            "{'x': 1, 'y': 2}",
        ),
        Pair(
            "collection/map_operation",
            "KARN interpreter conformance",
            source("print([x * 2 for x in [1, 2, 3]])"),
            source("! [1, 2, 3]*(x -> x * 2)"),
            "[2, 4, 6]",
        ),
        Pair(
            "collection/filter_operation",
            "KARN interpreter conformance",
            source("print([x for x in [1, 2, 3, 4] if x > 2])"),
            source("! [1, 2, 3, 4]%(x -> x > 2)"),
            "[3, 4]",
        ),
        Pair(
            "result/construct_ok",
            "KARN interpreter conformance",
            source('result = ("Ok", 42)\nprint(f"{result[0]}({result[1]})")'),
            source("! Ok(42)"),
            "Ok(42)",
        ),
        Pair(
            "result/propagate",
            "KARN interpreter conformance",
            source('result = ("Ok", 42)\nprint(result[1])'),
            source("! Ok(42)?"),
            "42",
        ),
        Pair(
            "result/fallback",
            "KARN interpreter conformance",
            source(
                'result = ("Err", "fail")\n'
                'print(42 if result[0] == "Err" else result[1])'
            ),
            source('! Err("fail")??42'),
            "42",
        ),
        Pair(
            "record/field",
            "KARN interpreter conformance",
            source('point = {"x": 10, "y": 20}\nprint(point["x"])'),
            source("type Pt:{x:N, y:N}\np = Pt(10, 20)\n! p.x"),
            "10",
        ),
        Pair(
            "match/ok",
            "KARN interpreter conformance",
            source(
                'value = ("Ok", 42)\n'
                'print(value[1] if value[0] == "Ok" else 0)'
            ),
            source("! match Ok(42){ Ok(v) -> v, Err(e) -> 0 }"),
            "42",
        ),
        Pair(
            "match/err",
            "KARN interpreter conformance",
            source(
                'value = ("Err", "fail")\n'
                'print(value[1] if value[0] == "Ok" else 0)'
            ),
            source('! match Err("fail"){ Ok(v) -> v, Err(e) -> 0 }'),
            "0",
        ),
        Pair(
            "pipeline/double_twice",
            "KARN interpreter conformance",
            source(
                """
                def double(x):
                    return x * 2

                print(double(double(5)))
                """
            ),
            source("double->x: x*2\n! double(5) |> double"),
            "20",
        ),
        Pair(
            "recursion/factorial",
            "KARN interpreter conformance",
            source(
                """
                def fact(n):
                    if n == 0:
                        return 1
                    return n * fact(n - 1)

                print(fact(5))
                """
            ),
            source(
                """
                fact->n:N:N
                  match n{ 0 -> 1, _ -> n * fact(n - 1) }
                ! fact(5)
                """
            ),
            "120",
        ),
        Pair(
            "string/upper",
            "KARN interpreter conformance",
            source('print("hello".upper())'),
            source('! "hello".upper()'),
            "HELLO",
        ),
        Pair(
            "collection/len",
            "KARN interpreter conformance",
            source("print(len([1, 2, 3]))"),
            source("! [1, 2, 3].len()"),
            "3",
        ),
        Pair(
            "collection/first",
            "KARN interpreter conformance",
            source("print([10, 20, 30][0])"),
            source("! [10, 20, 30].first()"),
            "10",
        ),
        Pair(
            "collection/spread_list",
            "KARN interpreter conformance",
            source("a = [1, 2]\nprint([*a, 3])"),
            source("a = [1, 2]\n! [*a, 3]"),
            "[1, 2, 3]",
        ),
        Pair(
            "collection/spread_map",
            "KARN interpreter conformance",
            source('a = {"x": 1}\nprint({**a, "y": 2})'),
            source("a = {x:1}\n! {*a, y:2}"),
            "{'x': 1, 'y': 2}",
        ),
        Pair(
            "function/lambda",
            "KARN interpreter conformance",
            source("square = lambda x: x * x\nprint(square(5))"),
            source("square = x -> x * x\n! square(5)"),
            "25",
        ),
        Pair(
            "stdlib/json",
            "KARN interpreter conformance",
            source(
                """
                import json

                print(json.loads('{"x":1}'))
                """
            ),
            source('! json.parse(\'{"x":1}\')?'),
            "{'x': 1}",
        ),
        Pair(
            "stdlib/sqrt",
            "KARN interpreter conformance",
            source("import math\nprint(math.sqrt(16))"),
            source("! math.sqrt(16)"),
            "4.0",
        ),
        Pair(
            "stdlib/md5",
            "KARN interpreter conformance",
            source(
                """
                import hashlib

                print(hashlib.md5("hello".encode()).hexdigest())
                """
            ),
            source('! crypto.md5("hello")?'),
            "5d41402abc4b2a76b9719d911017c592",
        ),
        Pair(
            "stdlib/join",
            "KARN interpreter conformance",
            source('print("-".join(["a", "b", "c"]))'),
            source('! str.join(["a", "b", "c"], "-")?'),
            "a-b-c",
        ),
        Pair(
            "arithmetic/modulo",
            "KARN interpreter conformance",
            source("print(29 % 5)"),
            source("! 29 % 5"),
            "4",
        ),
        Pair(
            "recursion/sum",
            "KARN extended public-feature probe",
            source(
                """
                def total(n):
                    if n == 0:
                        return 0
                    return n + total(n - 1)

                print(total(10))
                """
            ),
            source(
                """
                sum->n:N:N
                  match n{ 0 -> 0, _ -> n + sum(n - 1) }
                ! sum(10)
                """
            ),
            "55",
        ),
        Pair(
            "recursion/gcd",
            "KARN extended public-feature probe",
            source(
                """
                def gcd(a, b):
                    if b == 0:
                        return a
                    return gcd(b, a % b)

                print(gcd(48, 18))
                """
            ),
            source(
                """
                gcd->a:N b:N:N
                  match b{ 0 -> a, _ -> gcd(b, a % b) }
                ! gcd(48, 18)
                """
            ),
            "6",
        ),
        Pair(
            "collection/map_filter_chain",
            "KARN extended public-feature probe",
            source("print([x for x in [x * x for x in [1, 2, 3, 4]] if x > 4])"),
            source("! [1,2,3,4]*(x->x*x)%(x->x>4)"),
            "[9, 16]",
        ),
        Pair(
            "string/split",
            "KARN extended public-feature probe",
            source('print("a,b,c".split(","))'),
            source('! "a,b,c".split(",")'),
            "['a', 'b', 'c']",
        ),
        Pair(
            "string/contains",
            "KARN extended public-feature probe",
            source('print("er" in "kern")'),
            source('! "kern".contains("er")'),
            "True",
        ),
        Pair(
            "collection/last",
            "KARN extended public-feature probe",
            source("print([10, 20, 30][-1])"),
            source("! [10,20,30].last()"),
            "30",
        ),
        Pair(
            "stdlib/pow",
            "KARN extended public-feature probe",
            source("import math\nprint(math.pow(2, 8))"),
            source("! math.pow(2,8)"),
            "256.0",
        ),
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} pairs, found {len(pairs)}")
    ids = [pair.task_id for pair in pairs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("KARN pair task IDs must be unique.")
    return pairs


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def strip_karn_comments(value: str) -> str:
    return (
        "\n".join(
            line
            for line in value.splitlines()
            if not line.lstrip().startswith("--")
        ).strip()
        + "\n"
    )


def run_command(command: list[str]) -> tuple[bool, str, str]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = ANSI_ESCAPE.sub("", completed.stdout).strip()
    stderr = ANSI_ESCAPE.sub("", completed.stderr).strip()
    return completed.returncode == 0, stdout, stderr


def score_pairs(
    *,
    pairs: list[Pair],
    karn_root: Path,
    tokenizer: Tokenizer,
) -> list[PairResult]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    karn_cli = karn_root / "files" / "karn.py"
    results: list[PairResult] = []
    with tempfile.TemporaryDirectory(prefix="kern-karn-pairs-") as directory:
        temp = Path(directory)
        for index, pair in enumerate(pairs):
            python_source = pair.python.strip()
            karn_source = pair.karn.strip()
            compact_tree_value = compact_tree(ast.parse(python_source))
            expected_compact = ast.unparse(compact_tree_value)
            kern_source = transpile(python_source, compact=True).strip()
            decoded = compile_kern(kern_source)
            minified = python_minifier.minify(
                python_source,
                rename_globals=False,
            )
            kern_contract = (
                normalize_ast(decoded) == normalize_ast(expected_compact)
            )

            native_ids = tokenizer.encode(kern_source).ids
            native_exact = tokenizer.decode(native_ids) == kern_source

            python_path = temp / f"{index:03d}-python.py"
            kern_path = temp / f"{index:03d}-kern.py"
            minified_path = temp / f"{index:03d}-minified.py"
            karn_path = temp / f"{index:03d}.kn"
            generated_path = temp / f"{index:03d}-karn-generated.py"
            python_path.write_text(python_source + "\n", encoding="utf-8")
            kern_path.write_text(decoded + "\n", encoding="utf-8")
            minified_path.write_text(minified + "\n", encoding="utf-8")
            karn_path.write_text(karn_source + "\n", encoding="utf-8")

            python_ok, python_stdout, _ = run_command(
                [sys.executable, str(python_path)]
            )
            kern_ok, kern_stdout, _ = run_command(
                [sys.executable, str(kern_path)]
            )
            minified_ok, minified_stdout, _ = run_command(
                [sys.executable, str(minified_path)]
            )
            check_ok, _, check_error = run_command(
                [sys.executable, str(karn_cli), "check", str(karn_path)]
            )
            interpreter_ok, karn_stdout, interpreter_error = run_command(
                [sys.executable, str(karn_cli), "run", str(karn_path)]
            )
            codegen_ok, _, codegen_error = run_command(
                [
                    sys.executable,
                    str(karn_cli),
                    "build",
                    str(karn_path),
                    "--target",
                    "python",
                    "--out",
                    str(generated_path),
                ]
            )
            generated_ok = False
            generated_stdout = ""
            generated_error = ""
            if codegen_ok and generated_path.exists():
                generated_ok, generated_stdout, generated_error = run_command(
                    [sys.executable, str(generated_path)]
                )

            error_stage = ""
            error_message = ""
            if not check_ok:
                error_stage = "check"
                error_message = check_error
            elif not interpreter_ok:
                error_stage = "interpreter"
                error_message = interpreter_error
            elif not codegen_ok:
                error_stage = "python_codegen"
                error_message = codegen_error
            elif not generated_ok:
                error_stage = "generated_python"
                error_message = generated_error
            elif generated_stdout != pair.expected_stdout:
                error_stage = "generated_python_oracle"
                error_message = (
                    f"expected {pair.expected_stdout!r}, "
                    f"got {generated_stdout!r}"
                )

            results.append(
                PairResult(
                    task_id=pair.task_id,
                    origin=pair.origin,
                    python_sha256=sha256_text(python_source),
                    karn_sha256=sha256_text(karn_source),
                    python_cl100k=len(encodings["cl100k_base"].encode(python_source)),
                    kern_cl100k=len(encodings["cl100k_base"].encode(kern_source)),
                    python_minifier_cl100k=len(
                        encodings["cl100k_base"].encode(minified)
                    ),
                    karn_cl100k=len(encodings["cl100k_base"].encode(karn_source)),
                    python_o200k=len(encodings["o200k_base"].encode(python_source)),
                    kern_o200k=len(encodings["o200k_base"].encode(kern_source)),
                    python_minifier_o200k=len(
                        encodings["o200k_base"].encode(minified)
                    ),
                    karn_o200k=len(encodings["o200k_base"].encode(karn_source)),
                    kern_native_16k=len(native_ids),
                    kern_native_exact_roundtrip=native_exact,
                    kern_contract_ast=kern_contract,
                    python_oracle_ok=(
                        python_ok and python_stdout == pair.expected_stdout
                    ),
                    kern_oracle_ok=(
                        kern_ok and kern_stdout == pair.expected_stdout
                    ),
                    python_minifier_oracle_ok=(
                        minified_ok
                        and minified_stdout == pair.expected_stdout
                    ),
                    karn_check_ok=check_ok,
                    karn_interpreter_oracle_ok=(
                        interpreter_ok
                        and karn_stdout == pair.expected_stdout
                    ),
                    karn_python_codegen_ok=codegen_ok,
                    karn_python_codegen_oracle_ok=(
                        generated_ok
                        and generated_stdout == pair.expected_stdout
                    ),
                    karn_error_stage=error_stage,
                    karn_error_message=error_message[-1_000:],
                )
            )
    return results


def pct_below(value: int, baseline: int) -> float:
    return (baseline - value) / baseline * 100 if baseline else 0.0


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    totals = {
        "programs": len(results),
        "cl100k_base": {
            "python": sum(result.python_cl100k for result in results),
            "kern_compact": sum(result.kern_cl100k for result in results),
            "python_minifier": sum(
                result.python_minifier_cl100k for result in results
            ),
            "karn": sum(result.karn_cl100k for result in results),
        },
        "o200k_base": {
            "python": sum(result.python_o200k for result in results),
            "kern_compact": sum(result.kern_o200k for result in results),
            "python_minifier": sum(
                result.python_minifier_o200k for result in results
            ),
            "karn": sum(result.karn_o200k for result in results),
        },
        "native_system": {
            "kern_native_16k": sum(
                result.kern_native_16k for result in results
            ),
            "karn_cl100k_base": sum(
                result.karn_cl100k for result in results
            ),
        },
        "functional": {
            "python": sum(result.python_oracle_ok for result in results),
            "kern_compact": sum(result.kern_oracle_ok for result in results),
            "python_minifier": sum(
                result.python_minifier_oracle_ok for result in results
            ),
            "karn_check": sum(result.karn_check_ok for result in results),
            "karn_interpreter": sum(
                result.karn_interpreter_oracle_ok for result in results
            ),
            "karn_python_codegen_produced": sum(
                result.karn_python_codegen_ok for result in results
            ),
            "karn_python_codegen": sum(
                result.karn_python_codegen_oracle_ok for result in results
            ),
        },
        "structural": {
            "kern_contract_ast": sum(
                result.kern_contract_ast for result in results
            ),
            "kern_native_exact_roundtrip": sum(
                result.kern_native_exact_roundtrip for result in results
            ),
        },
    }
    cl = totals["cl100k_base"]
    native = totals["native_system"]
    totals["comparisons"] = {
        "kern_cl100k_below_karn_pct": pct_below(
            cl["kern_compact"], cl["karn"]
        ),
        "kern_native_below_karn_cl100k_pct": pct_below(
            native["kern_native_16k"], native["karn_cl100k_base"]
        ),
        "kern_shared_wins": sum(
            result.kern_cl100k < result.karn_cl100k
            for result in results
        ),
        "kern_shared_ties": sum(
            result.kern_cl100k == result.karn_cl100k
            for result in results
        ),
        "kern_native_system_wins": sum(
            result.kern_native_16k < result.karn_cl100k
            for result in results
        ),
        "median_per_pair_kern_cl100k_below_karn_pct": statistics.median(
            pct_below(result.kern_cl100k, result.karn_cl100k)
            for result in results
        ),
    }
    return totals


def claim_audit(karn_root: Path) -> dict[str, Any]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    api_path = karn_root / "examples" / "api-server.kn"
    api_source = strip_karn_comments(api_path.read_text(encoding="utf-8")).strip()
    snippet = source(PUBLIC_API_SNIPPET).strip()
    docs = (karn_root / "docs.html").read_text(encoding="utf-8")
    searchable = [
        path
        for path in karn_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    text_by_path: dict[str, str] = {}
    for path in searchable:
        try:
            text_by_path[str(path.relative_to(karn_root))] = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            continue
    count_47_hits = sorted(
        path for path, value in text_by_path.items() if "~47 tokens" in value
    )
    count_198_hits = sorted(
        path for path, value in text_by_path.items() if "~198 tokens" in value
    )
    tokenizer_hits = sorted(
        path
        for path, value in text_by_path.items()
        if "cl100k" in value.lower() or "tiktoken" in value.lower()
    )
    counting_script_hits = sorted(
        path
        for path, value in text_by_path.items()
        if (
            "count_tokens" in value
            or "token_count" in value
            or "encode_ordinary" in value
        )
    )
    return {
        "published_claim": {
            "description": (
                "Same REST API with auth, DB query, and JSON response"
            ),
            "karn_approximate_tokens": 47,
            "python_approximate_tokens": 198,
            "claimed_reduction_pct": 76,
            "tokenizer_named": bool(tokenizer_hits),
            "paired_sources_published": (
                karn_root.joinpath("examples/api-server.python.py").exists()
            ),
            "source_search": (
                "47 and 198 appear only as approximate HTML values; no "
                "paired Python/TypeScript/Rust sources or counting script"
            ),
            "approximate_47_hits": count_47_hits,
            "approximate_198_hits": count_198_hits,
            "tokenizer_reference_hits": tokenizer_hits,
            "counting_script_hits": counting_script_hits,
            "claim_present": (
                "~47 tokens" in docs
                and "~198 tokens" in docs
                and "76% fewer tokens" in docs
            ),
        },
        "current_public_readme_snippet": {
            "sha256": sha256_text(snippet),
            "cl100k_base": len(encodings["cl100k_base"].encode(snippet)),
            "o200k_base": len(encodings["o200k_base"].encode(snippet)),
        },
        "current_public_api_server_file": {
            "path": "examples/api-server.kn",
            "comments_excluded": True,
            "sha256": sha256_text(api_source),
            "cl100k_base": len(encodings["cl100k_base"].encode(api_source)),
            "o200k_base": len(encodings["o200k_base"].encode(api_source)),
        },
    }


def official_gates(karn_root: Path) -> dict[str, Any]:
    tests_ok, tests_stdout, tests_stderr = run_command(
        [sys.executable, str(karn_root / "tests" / "test_karn.py")]
    )
    examples = sorted((karn_root / "examples").glob("*.kn"))
    check_ok, check_stdout, check_stderr = run_command(
        [
            sys.executable,
            str(karn_root / "files" / "karn.py"),
            "check",
            *map(str, examples),
        ]
    )
    return {
        "official_tests": {
            "ok": tests_ok,
            "reported_passes": 91 if "91 passed, 0 failed" in tests_stdout else 0,
            "error": tests_stderr[-1_000:],
        },
        "official_examples_check": {
            "ok": check_ok,
            "checked": check_stdout.count(" — OK ("),
            "total": len(examples),
            "implementation_note": (
                "KARN check_file invokes Lexer and Parser only; despite the "
                "CLI description, no separate type checker is called"
            ),
            "error": check_stderr[-1_000:],
        },
    }


def origin_aggregates(results: list[PairResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in sorted({result.origin for result in results}):
        subset = [result for result in results if result.origin == origin]
        rows.append(
            {
                "origin": origin,
                "programs": len(subset),
                "kern_cl100k": sum(
                    result.kern_cl100k for result in subset
                ),
                "karn_cl100k": sum(
                    result.karn_cl100k for result in subset
                ),
                "kern_native_16k": sum(
                    result.kern_native_16k for result in subset
                ),
                "kern_cl100k_below_karn_pct": pct_below(
                    sum(result.kern_cl100k for result in subset),
                    sum(result.karn_cl100k for result in subset),
                ),
                "kern_native_below_karn_cl100k_pct": pct_below(
                    sum(result.kern_native_16k for result in subset),
                    sum(result.karn_cl100k for result in subset),
                ),
                "karn_interpreter_oracle": sum(
                    result.karn_interpreter_oracle_ok
                    for result in subset
                ),
                "karn_python_codegen_oracle": sum(
                    result.karn_python_codegen_oracle_ok
                    for result in subset
                ),
            }
        )
    return rows


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
        ("KARN", values["karn"], "#f59e0b"),
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
        "<title id=\"title\">Kern versus KARN paired token density</title>",
        (
            "<desc id=\"desc\">Aggregate cl100k_base tokens on 46 matched "
            "executable programs; lower is better</desc>"
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            'font-weight="700">Kern versus KARN: shared tokenizer</text>'
        ),
        (
            f'<text x="{left}" y="66" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            "46 matched executable programs; lower is better</text>"
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
        title="Paired functional preservation",
        subtitle=(
            "Exact deterministic stdout oracle on all 46 programs; "
            "KARN interpreter and Python target shown separately"
        ),
        groups=["46 matched programs"],
        series=[
            (
                "Kern compact",
                "#22c55e",
                [functional["kern_compact"] / total * 100],
            ),
            (
                "KARN interpreter",
                "#f59e0b",
                [functional["karn_interpreter"] / total * 100],
            ),
            (
                "KARN Python target",
                "#ef4444",
                [functional["karn_python_codegen"] / total * 100],
            ),
        ],
        y_label="Exact oracle pass rate (%)",
        max_value=100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--karn-root",
        type=Path,
        required=True,
        help="Pinned KARN source checkout.",
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
        default=Path("benchmark_results/karn"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    actual_commit = git_commit(args.karn_root)
    if actual_commit != KARN_COMMIT:
        raise RuntimeError(
            f"KARN checkout drifted: {actual_commit} != {KARN_COMMIT}"
        )
    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern tokenizer SHA-256 does not match manifest.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    pairs = paired_programs()
    results = score_pairs(
        pairs=pairs,
        karn_root=args.karn_root,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "46 fixed matched programs derived from KARN public examples "
                "and conformance tests; full denominator; deterministic "
                "stdout; same production tokenizer in shared lanes"
            ),
            "karn_commit": actual_commit,
            "karn_version": KARN_VERSION,
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "claim_audit": claim_audit(args.karn_root),
        "official_gates": official_gates(args.karn_root),
        "paired_corpus": aggregate_row,
        "origin_aggregates": origin_aggregates(results),
        "failure_stages": {
            stage: sum(result.karn_error_stage == stage for result in results)
            for stage in sorted(
                {
                    result.karn_error_stage
                    for result in results
                    if result.karn_error_stage
                }
            )
        },
    }
    summary_path = args.output_dir / "karn-benchmark-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "karn-pair-corpus.json").write_text(
        json.dumps(
            [asdict(pair) for pair in pairs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_details(results, args.output_dir / "karn-benchmark-details.csv")
    write_token_svg(args.output_dir / "karn-token-density.svg", aggregate_row)
    write_functional_svg(
        args.output_dir / "karn-functional-preservation.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
