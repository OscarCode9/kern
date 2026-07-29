"""Executable token-density screen for Kern, K, GolfScript, and J.

The public code.golf leaderboard is a useful discovery signal, but its leading
solutions are private and it scores UTF-8 bytes rather than LLM tokens.  This
harness therefore does not pretend to reproduce those private solutions.  It
publishes a fixed, inspectable corpus of equivalent programs, executes every
language, and counts every source under the same production tokenizers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
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

from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile

K_COMMIT = "717063f24921d5aff405a39cf7643efedb5bb365"
GOLFSCRIPT_COMMIT = "6155e9f7860775be53bdc79c6e1c3b9308ebbfe5"
GOLFSCRIPT_SHA256 = (
    "c3d9800af812146c0215a8a61aa5fee615ccdb1bed3a3ff5f64b8b4e0a28c25e"
)
J_VERSION = "9.6.3"
CODE_GOLF_COMMIT = "2c0fc35ca0f76a2a6c7faaf4d32f21244a359a95"
EXPECTED_PAIRS = 14


def sha256_text(value: str) -> str:
    """Return a stable digest for an in-memory source string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a runtime artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ast(source_text: str) -> str:
    """Return a location-free AST representation for contract comparisons."""
    return ast.dump(ast.parse(source_text), include_attributes=False)


def write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    y_label: str,
    max_value: float = 100.0,
    value_suffix: str = "%",
) -> None:
    """Write a dependency-free grouped bar chart."""
    width, height = 980, 540
    left, right, top, bottom = 92, 34, 92, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / max(1, len(groups))
    bar_w = min(68, group_w / (len(series) + 1))
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        f'<title id="title">{svg_escape(title)}</title>',
        f'<desc id="desc">{svg_escape(subtitle)}</desc>',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            f'font-weight="700">{svg_escape(title)}</text>'
        ),
        (
            f'<text x="{left}" y="65" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            f'{svg_escape(subtitle)}</text>'
        ),
    ]
    tick_step = 20 if max_value >= 80 else 10
    for tick in range(0, int(max_value) + 1, tick_step):
        y = top + plot_h - tick / max_value * plot_h
        elements.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.1f}" '
                    f'x2="{left + plot_w}" y2="{y:.1f}" '
                    'stroke="#25314d" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 12}" y="{y + 5:.1f}" '
                    'fill="#94a3b8" text-anchor="end" '
                    'font-family="Inter,system-ui,sans-serif" font-size="12">'
                    f'{tick}</text>'
                ),
            ]
        )
    elements.append(
        f'<text x="22" y="{top + plot_h / 2:.1f}" fill="#94a3b8" '
        f'text-anchor="middle" transform="rotate(-90 22 '
        f'{top + plot_h / 2:.1f})" '
        'font-family="Inter,system-ui,sans-serif" font-size="13">'
        f'{svg_escape(y_label)}</text>'
    )
    for group_index, group in enumerate(groups):
        center = left + group_w * (group_index + 0.5)
        start = center - bar_w * len(series) / 2
        elements.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 28}" '
            'fill="#cbd5e1" text-anchor="middle" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            f'{svg_escape(group)}</text>'
        )
        for series_index, (_, color, values) in enumerate(series):
            value = values[group_index]
            bar_h = max(0.0, min(max_value, value)) / max_value * plot_h
            x = start + series_index * bar_w + 4
            y = top + plot_h - bar_h
            elements.extend(
                [
                    (
                        f'<rect x="{x:.1f}" y="{y:.1f}" '
                        f'width="{bar_w - 8:.1f}" height="{bar_h:.1f}" '
                        f'rx="5" fill="{color}"/>'
                    ),
                    (
                        f'<text x="{x + (bar_w - 8) / 2:.1f}" '
                        f'y="{max(top + 14, y - 8):.1f}" fill="#f8fafc" '
                        'text-anchor="middle" '
                        'font-family="Inter,system-ui,sans-serif" '
                        f'font-size="12" font-weight="600">{value:.1f}'
                        f'{svg_escape(value_suffix)}</text>'
                    ),
                ]
            )
    legend_x = left
    legend_y = height - 24
    for label, color, _ in series:
        elements.extend(
            [
                (
                    f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" '
                    f'height="14" rx="3" fill="{color}"/>'
                ),
                (
                    f'<text x="{legend_x + 22}" y="{legend_y}" '
                    'fill="#cbd5e1" '
                    'font-family="Inter,system-ui,sans-serif" font-size="13">'
                    f'{svg_escape(label)}</text>'
                ),
            ]
        )
        legend_x += 190
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Pair:
    task_id: str
    category: str
    python: str
    k: str
    golfscript: str
    j: str
    expected_stdout: str


@dataclass
class PairResult:
    task_id: str
    category: str
    python_sha256: str
    k_sha256: str
    golfscript_sha256: str
    j_sha256: str
    python_bytes: int
    kern_bytes: int
    python_minifier_bytes: int
    k_bytes: int
    golfscript_bytes: int
    j_bytes: int
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    k_cl100k: int
    golfscript_cl100k: int
    j_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    k_o200k: int
    golfscript_o200k: int
    j_o200k: int
    kern_native_16k: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    k_oracle_ok: bool
    golfscript_oracle_ok: bool
    j_oracle_ok: bool
    k_error: str
    golfscript_error: str
    j_error: str


def source(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


def paired_programs() -> list[Pair]:
    pairs = [
        Pair(
            "scalar/arithmetic",
            "scalar",
            source(
                """
                print((17 * 23 + 11) // 2)
                """
            ),
            "`0:$-2!((17*23)+11)\n",
            "17 23*11+2/\n",
            "echo ((17*23)+11)%2\n",
            "201",
        ),
        Pair(
            "reduction/sum_1_100",
            "reduction",
            source(
                """
                print(sum(range(1, 101)))
                """
            ),
            "`0:$+/1+!100\n",
            "100,{)}%{+}*\n",
            "echo +/1+i.100\n",
            "5050",
        ),
        Pair(
            "reduction/factorial_10",
            "reduction",
            source(
                """
                import math
                print(math.factorial(10))
                """
            ),
            "`0:$*/1+!10\n",
            "10 1+,1>{*}*\n",
            "echo */1+i.10\n",
            "3628800",
        ),
        Pair(
            "text/reverse",
            "text",
            source(
                """
                print("kernlanguage"[::-1])
                """
            ),
            '`0:|"kernlanguage"\n',
            "'kernlanguage'-1%\n",
            "echo |.'kernlanguage'\n",
            "egaugnalnrek",
        ),
        Pair(
            "array/sort",
            "array",
            source(
                """
                print(*sorted([9, 1, 5, 3, 7, 2, 8, 6, 4]))
                """
            ),
            "`0:$a@<a:a:9 1 5 3 7 2 8 6 4\n",
            "[9 1 5 3 7 2 8 6 4]$' '*\n",
            "echo /:~9 1 5 3 7 2 8 6 4\n",
            "1 2 3 4 5 6 7 8 9",
        ),
        Pair(
            "array/distinct",
            "array",
            source(
                """
                print(*dict.fromkeys([3, 1, 2, 3, 2, 4, 1, 5]))
                """
            ),
            "`0:$?3 1 2 3 2 4 1 5\n",
            "[3 1 2 3 2 4 1 5][]|' '*\n",
            "echo ~.3 1 2 3 2 4 1 5\n",
            "3 1 2 4 5",
        ),
        Pair(
            "array/squares",
            "array",
            source(
                """
                print(*(x * x for x in range(1, 11)))
                """
            ),
            "`0:$a*a:a:1+!10\n",
            "10,{).*}%' '*\n",
            "echo *:1+i.10\n",
            "1 4 9 16 25 36 49 64 81 100",
        ),
        Pair(
            "array/evens",
            "array",
            source(
                """
                print(*(x for x in range(1, 21) if x % 2 == 0))
                """
            ),
            "`0:$a@&~2!a:a:1+!20\n",
            "20,{)}%{2%!},' '*\n",
            "echo (#~0=2&|)1+i.20\n",
            "2 4 6 8 10 12 14 16 18 20",
        ),
        Pair(
            "text/count_character",
            "text",
            source(
                """
                print("abracadabra".count("a"))
                """
            ),
            '`0:$+/"abracadabra"="a"\n',
            "'abracadabra'{97=},,\n",
            "echo +/'abracadabra'='a'\n",
            "5",
        ),
        Pair(
            "array/dot_product",
            "array",
            source(
                """
                print(sum(a * b for a, b in zip([1, 2, 3], [4, 5, 6])))
                """
            ),
            "`0:$+/1 2 3*4 5 6\n",
            "[[1 2 3][4 5 6]]zip{{*}*}%{+}*\n",
            "echo +/1 2 3*4 5 6\n",
            "32",
        ),
        Pair(
            "text/palindrome",
            "text",
            source(
                """
                text = "racecar"
                print(int(text == text[::-1]))
                """
            ),
            '`0:$"racecar"~|"racecar"\n',
            "'racecar'.-1%=\n",
            "echo 'racecar'-:|.'racecar'\n",
            "1",
        ),
        Pair(
            "scalar/gcd",
            "scalar",
            source(
                """
                import math
                print(math.gcd(2706, 410))
                """
            ),
            "gcd:*|(*:)(|!\\)/,;`0:$gcd[2706]410\n",
            "2706 410{.@\\%.}do;\n",
            "echo 2706+.410\n",
            "82",
        ),
        Pair(
            "array/rotate_left",
            "array",
            source(
                """
                values = [1, 2, 3, 4, 5]
                print(*(values[3:] + values[:3]))
                """
            ),
            "`0:$5#3_(a,a:1 2 3 4 5)\n",
            "[1 2 3 4 5].3>\\3<+' '*\n",
            "echo 3|.1 2 3 4 5\n",
            "4 5 1 2 3",
        ),
        Pair(
            "recurrence/fibonacci",
            "recurrence",
            source(
                """
                values = [0, 1]
                for _ in range(10):
                    values.append(values[-1] + values[-2])
                print(*values)
                """
            ),
            "`0:${x,+/-2#x}/[10;0 1]\n",
            "0 1{100<}{.@+}/\\;[0]\\+' '*\n",
            "echo (3 : 'y,+/_2{.y')^:10]0 1\n",
            "0 1 1 2 3 5 8 13 21 34 55 89",
        ),
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} compact-language pairs.")
    if len({pair.task_id for pair in pairs}) != EXPECTED_PAIRS:
        raise RuntimeError("Compact-language pair IDs must be unique.")
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
    """Ignore display-only whitespace while preserving token content/order."""
    return " ".join(value.strip().split())


def pct_below(value: int, baseline: int) -> float:
    return (baseline - value) / baseline * 100 if baseline else 0.0


def score_pairs(
    *,
    pairs: list[Pair],
    k_binary: Path,
    golfscript: Path,
    j_binary: Path,
    ruby: str,
    tokenizer: Tokenizer,
) -> list[PairResult]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    results: list[PairResult] = []
    with tempfile.TemporaryDirectory(
        prefix="kern-compact-language-pairs-"
    ) as directory:
        temp = Path(directory)
        for index, pair in enumerate(pairs):
            python_source = pair.python.strip()
            compact_tree_value = compact_tree(ast.parse(python_source))
            expected_compact = ast.unparse(compact_tree_value)
            kern_source = transpile(python_source, compact=True).strip()
            decoded = compile_kern(kern_source)
            minified = python_minifier.minify(
                python_source,
                rename_globals=False,
            )
            sources = {
                "python": python_source,
                "kern": kern_source,
                "python_minifier": minified,
                "k": pair.k.strip(),
                "golfscript": pair.golfscript.strip(),
                "j": pair.j.strip(),
            }
            native_ids = tokenizer.encode(kern_source).ids

            python_path = temp / f"{index:02d}-python.py"
            kern_path = temp / f"{index:02d}-kern.py"
            minified_path = temp / f"{index:02d}-minified.py"
            k_path = temp / f"{index:02d}.k"
            golfscript_path = temp / f"{index:02d}.gs"
            j_path = temp / f"{index:02d}.ijs"
            for path, value in (
                (python_path, python_source),
                (kern_path, decoded),
                (minified_path, minified),
                (k_path, pair.k.strip()),
                (golfscript_path, pair.golfscript.strip()),
                (j_path, pair.j.strip()),
            ):
                path.write_text(value + "\n", encoding="utf-8")

            python_ok, python_stdout, _ = run_command(
                [sys.executable, str(python_path)]
            )
            kern_ok, kern_stdout, _ = run_command(
                [sys.executable, str(kern_path)]
            )
            minifier_ok, minifier_stdout, _ = run_command(
                [sys.executable, str(minified_path)]
            )
            k_ok, k_stdout, k_error = run_command(
                [str(k_binary), str(k_path)]
            )
            golfscript_ok, golfscript_stdout, golfscript_error = run_command(
                [
                    ruby,
                    "--encoding",
                    "ASCII-8BIT",
                    str(golfscript),
                    str(golfscript_path),
                ]
            )
            j_ok, j_stdout, j_error = run_command(
                [str(j_binary), str(j_path)]
            )
            expected = normalize_stdout(pair.expected_stdout)

            cl = {
                name: len(encodings["cl100k_base"].encode(value))
                for name, value in sources.items()
            }
            o = {
                name: len(encodings["o200k_base"].encode(value))
                for name, value in sources.items()
            }
            byte_counts = {
                name: len(value.encode("utf-8"))
                for name, value in sources.items()
            }
            results.append(
                PairResult(
                    task_id=pair.task_id,
                    category=pair.category,
                    python_sha256=sha256_text(python_source),
                    k_sha256=sha256_text(pair.k.strip()),
                    golfscript_sha256=sha256_text(pair.golfscript.strip()),
                    j_sha256=sha256_text(pair.j.strip()),
                    python_bytes=byte_counts["python"],
                    kern_bytes=byte_counts["kern"],
                    python_minifier_bytes=byte_counts["python_minifier"],
                    k_bytes=byte_counts["k"],
                    golfscript_bytes=byte_counts["golfscript"],
                    j_bytes=byte_counts["j"],
                    python_cl100k=cl["python"],
                    kern_cl100k=cl["kern"],
                    python_minifier_cl100k=cl["python_minifier"],
                    k_cl100k=cl["k"],
                    golfscript_cl100k=cl["golfscript"],
                    j_cl100k=cl["j"],
                    python_o200k=o["python"],
                    kern_o200k=o["kern"],
                    python_minifier_o200k=o["python_minifier"],
                    k_o200k=o["k"],
                    golfscript_o200k=o["golfscript"],
                    j_o200k=o["j"],
                    kern_native_16k=len(native_ids),
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
                        kern_ok and normalize_stdout(kern_stdout) == expected
                    ),
                    python_minifier_oracle_ok=(
                        minifier_ok
                        and normalize_stdout(minifier_stdout) == expected
                    ),
                    k_oracle_ok=(
                        k_ok and normalize_stdout(k_stdout) == expected
                    ),
                    golfscript_oracle_ok=(
                        golfscript_ok
                        and normalize_stdout(golfscript_stdout) == expected
                    ),
                    j_oracle_ok=(
                        j_ok and normalize_stdout(j_stdout) == expected
                    ),
                    k_error=(
                        ""
                        if k_ok and normalize_stdout(k_stdout) == expected
                        else (k_error or k_stdout)[-1_000:]
                    ),
                    golfscript_error=(
                        ""
                        if golfscript_ok
                        and normalize_stdout(golfscript_stdout) == expected
                        else (golfscript_error or golfscript_stdout)[-1_000:]
                    ),
                    j_error=(
                        ""
                        if j_ok and normalize_stdout(j_stdout) == expected
                        else (j_error or j_stdout)[-1_000:]
                    ),
                )
            )
    return results


def aggregate(results: list[PairResult]) -> dict[str, Any]:
    representations = (
        "python",
        "kern",
        "python_minifier",
        "k",
        "golfscript",
        "j",
    )
    cl = {
        name: sum(getattr(result, f"{name}_cl100k") for result in results)
        for name in representations
    }
    o = {
        name: sum(getattr(result, f"{name}_o200k") for result in results)
        for name in representations
    }
    byte_totals = {
        name: sum(getattr(result, f"{name}_bytes") for result in results)
        for name in representations
    }
    functional = {
        "python": sum(result.python_oracle_ok for result in results),
        "kern": sum(result.kern_oracle_ok for result in results),
        "python_minifier": sum(
            result.python_minifier_oracle_ok for result in results
        ),
        "k": sum(result.k_oracle_ok for result in results),
        "golfscript": sum(result.golfscript_oracle_ok for result in results),
        "j": sum(result.j_oracle_ok for result in results),
    }
    native_kern = sum(result.kern_native_16k for result in results)
    competitors = ("k", "golfscript", "j")
    categories: dict[str, Any] = {}
    for category in sorted({result.category for result in results}):
        category_results = [
            result for result in results if result.category == category
        ]
        categories[category] = {
            "programs": len(category_results),
            "cl100k_base": {
                name: sum(
                    getattr(result, f"{name}_cl100k")
                    for result in category_results
                )
                for name in representations
            },
            "kern_native_16k": sum(
                result.kern_native_16k for result in category_results
            ),
        }
    return {
        "programs": len(results),
        "cl100k_base": cl,
        "o200k_base": o,
        "utf8_bytes": byte_totals,
        "native_system": {
            "kern_native_16k": native_kern,
            "k_cl100k_base": cl["k"],
            "golfscript_cl100k_base": cl["golfscript"],
            "j_cl100k_base": cl["j"],
        },
        "functional": functional,
        "structural": {
            "kern_contract_ast": sum(
                result.kern_contract_ast for result in results
            ),
            "kern_native_exact_roundtrip": sum(
                result.kern_native_exact_roundtrip for result in results
            ),
        },
        "comparisons": {
            "shared_kern_below_pct": {
                name: pct_below(cl["kern"], cl[name])
                for name in competitors
            },
            "native_kern_below_competitor_cl100k_pct": {
                name: pct_below(native_kern, cl[name])
                for name in competitors
            },
            "shared_kern_wins": {
                name: sum(
                    result.kern_cl100k < getattr(result, f"{name}_cl100k")
                    for result in results
                )
                for name in competitors
            },
            "native_kern_wins": {
                name: sum(
                    result.kern_native_16k
                    < getattr(result, f"{name}_cl100k")
                    for result in results
                )
                for name in competitors
            },
            "median_per_pair_shared_kern_below_pct": {
                name: statistics.median(
                    pct_below(
                        result.kern_cl100k,
                        getattr(result, f"{name}_cl100k"),
                    )
                    for result in results
                )
                for name in competitors
            },
        },
        "categories": categories,
    }


def runtime_gates(
    *,
    k_root: Path,
    golfscript: Path,
    j_binary: Path,
    ruby: str,
) -> dict[str, Any]:
    k_actual_commit = git_commit(k_root)
    golfscript_hash = sha256_file(golfscript)
    ruby_path = shutil.which(ruby)
    j_ok, j_stdout, j_error = run_command(
        [str(j_binary), "-js", "exit echo JVERSION"]
    )
    return {
        "k": {
            "ok": k_actual_commit.startswith(K_COMMIT),
            "commit": k_actual_commit,
            "expected_prefix": K_COMMIT,
            "binary_sha256": sha256_file(k_root / "k"),
        },
        "golfscript": {
            "ok": golfscript_hash == GOLFSCRIPT_SHA256,
            "commit": GOLFSCRIPT_COMMIT,
            "script_sha256": golfscript_hash,
            "expected_sha256": GOLFSCRIPT_SHA256,
            "ruby": ruby_path or ruby,
        },
        "j": {
            "ok": j_ok and f"Engine: j{J_VERSION}" in j_stdout,
            "version_output": j_stdout,
            "expected_version": J_VERSION,
            "binary_sha256": sha256_file(j_binary),
            "error": j_error,
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
        ("K", values["k"], "#a855f7"),
        ("GolfScript", values["golfscript"], "#f97316"),
        ("J", values["j"], "#f59e0b"),
        ("Kern compact", values["kern"], "#22c55e"),
    ]
    width, height = 1120, 600
    left, right, top, bottom = 92, 34, 100, 118
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
        '<title id="title">Compact-language token-density screen</title>',
        (
            '<desc id="desc">Aggregate cl100k_base tokens on fourteen matched '
            "executable programs; lower is better</desc>"
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            'font-weight="700">Kern versus the compact-language frontier</text>'
        ),
        (
            f'<text x="{left}" y="66" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            "14 matched executable programs · cl100k_base · lower is better"
            "</text>"
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
                    'font-family="Inter,system-ui,sans-serif" font-size="12">'
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
        title="Compact-language functional preservation",
        subtitle=(
            "Every representation is executed against the same normalized "
            "stdout oracle"
        ),
        groups=["14 matched programs"],
        series=[
            (
                "Kern",
                "#22c55e",
                [functional["kern"] / total * 100],
            ),
            ("K", "#a855f7", [functional["k"] / total * 100]),
            (
                "GolfScript",
                "#f97316",
                [functional["golfscript"] / total * 100],
            ),
            ("J", "#f59e0b", [functional["j"] / total * 100]),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def write_system_svg(path: Path, aggregate_row: dict[str, Any]) -> None:
    shared = aggregate_row["cl100k_base"]
    native = aggregate_row["native_system"]["kern_native_16k"]
    write_grouped_bar_svg(
        path,
        title="Native-system aggregate crosses the first frontier",
        subtitle=(
            "14 executable pairs · competitor sources use cl100k_base · "
            "lower is better"
        ),
        groups=["K", "GolfScript", "J"],
        series=[
            (
                "Competitor + cl100k",
                "#f59e0b",
                [shared["k"], shared["golfscript"], shared["j"]],
            ),
            (
                "Kern + cl100k",
                "#06b6d4",
                [shared["kern"], shared["kern"], shared["kern"]],
            ),
            (
                "Kern + Kern-16K",
                "#22c55e",
                [native, native, native],
            ),
        ],
        y_label="Aggregate tokens",
        max_value=230.0,
        value_suffix="",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k-root",
        type=Path,
        required=True,
        help="Pinned ngn/k source checkout with the `k` binary built.",
    )
    parser.add_argument(
        "--golfscript",
        type=Path,
        required=True,
        help="Pinned golfscript.rb script.",
    )
    parser.add_argument(
        "--j-binary",
        type=Path,
        required=True,
        help="J 9.6.3 jconsole binary.",
    )
    parser.add_argument("--ruby", default="ruby")
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
        default=Path("benchmark_results/compact-languages"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    k_binary = args.k_root / "k"
    for path in (k_binary, args.golfscript, args.j_binary):
        if not path.exists():
            raise RuntimeError(f"Required runtime artifact is missing: {path}")
    gates = runtime_gates(
        k_root=args.k_root,
        golfscript=args.golfscript,
        j_binary=args.j_binary,
        ruby=args.ruby,
    )
    failed_gates = [name for name, gate in gates.items() if not gate["ok"]]
    if failed_gates:
        raise RuntimeError(
            "Compact-language runtime gates failed: "
            + ", ".join(failed_gates)
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
        k_binary=k_binary,
        golfscript=args.golfscript,
        j_binary=args.j_binary,
        ruby=args.ruby,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    failed_oracles = {
        language: aggregate_row["programs"] - passed
        for language, passed in aggregate_row["functional"].items()
        if passed != aggregate_row["programs"]
    }
    if failed_oracles:
        raise RuntimeError(f"Compact-language oracle failures: {failed_oracles}")

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Fourteen fixed matched executable programs; complete public "
                "sources; shared production tokenizers; exact normalized "
                "stdout; separate native-system lane"
            ),
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "discovery_evidence": {
            "code_golf_repository_commit": CODE_GOLF_COMMIT,
            "ranking_snapshot_date": "2026-07-29",
            "all_hole_bytes": {
                "k": {"rank": 1, "bytes": 7755},
                "golfscript": {"rank": 2, "bytes": 8229},
                "j": {"rank": 3, "bytes": 8741},
            },
            "leader_sources_public": False,
            "interpretation": (
                "The ranking identifies high-risk languages but cannot be "
                "recounted as LLM tokens because leading sources are private"
            ),
        },
        "runtime_gates": gates,
        "corpus": {
            "authorship": (
                "Benchmark-authored compact programs using documented "
                "language primitives; not claimed to be globally minimal"
            ),
            "normalization": (
                "Collapse display-only whitespace; preserve value tokens and "
                "their order"
            ),
            "sources_and_hashes_published": True,
        },
        "results": aggregate_row,
    }
    (args.output_dir / "compact-language-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "compact-language-corpus.json").write_text(
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
        args.output_dir / "compact-language-details.csv",
    )
    write_token_svg(
        args.output_dir / "compact-language-token-density.svg",
        aggregate_row,
    )
    write_functional_svg(
        args.output_dir / "compact-language-functional.svg",
        aggregate_row,
    )
    write_system_svg(
        args.output_dir / "compact-language-native-system.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
