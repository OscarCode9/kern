"""Executable token-density screen for Kern, Pyth, and Jelly.

Pyth and Jelly are adversarial compact-language baselines, not Python source
minifiers.  This harness therefore uses a small fixed corpus of matched,
inspectable programs, executes every representation against the same stdout
oracle, and counts every complete source under the same production tokenizers.

Jelly's traditional one-byte code page is reported separately from UTF-8 bytes
and LLM tokens.  Those units are intentionally never combined.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import python_minifier
import tiktoken
from tokenizers import Tokenizer

from benchmark_compact_languages import (
    normalize_ast,
    normalize_stdout,
    paired_programs,
    pct_below,
    run_command,
    sha256_file,
    sha256_text,
    write_grouped_bar_svg,
)
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile

PYTH_COMMIT = "97cdf30d749d2a0d6ec1bb4b9bc417c34cce05bb"
JELLY_COMMIT = "70c9fd93ab009c05dc396f8cc091f72b212fb188"
PYTH_PYTHON_VERSION = "3.8.20"
EXPECTED_PAIRS = 14


@dataclass(frozen=True)
class GolfPair:
    task_id: str
    category: str
    python: str
    pyth: str
    jelly: str
    expected_stdout: str


@dataclass
class GolfResult:
    task_id: str
    category: str
    python_sha256: str
    kern_sha256: str
    python_minifier_sha256: str
    pyth_sha256: str
    jelly_sha256: str
    python_bytes: int
    kern_bytes: int
    python_minifier_bytes: int
    pyth_bytes: int
    jelly_bytes: int
    jelly_code_page_units: int
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    pyth_cl100k: int
    jelly_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    pyth_o200k: int
    jelly_o200k: int
    kern_native_16k: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    pyth_oracle_ok: bool
    jelly_oracle_ok: bool
    pyth_error: str
    jelly_error: str


def golf_pairs() -> list[GolfPair]:
    """Return the fixed matched-program registry."""
    compact_pairs = {pair.task_id: pair for pair in paired_programs()}
    competitors = {
        "scalar/arithmetic": (
            "/+*17 23 11 2",
            "17×23+11H",
        ),
        "reduction/sum_1_100": (
            "sS100",
            "100RS",
        ),
        "reduction/factorial_10": (
            ".!T",
            "10!",
        ),
        "text/reverse": (
            '_"kernlanguage"',
            "“kernlanguage”Ṛ",
        ),
        "array/sort": (
            "jdS[9 1 5 3 7 2 8 6 4)",
            "9,1,5,3,7,2,8,6,4ṢK",
        ),
        "array/distinct": (
            "jd{[3 1 2 3 2 4 1 5)",
            "3,1,2,3,2,4,1,5QK",
        ),
        "array/squares": (
            "jdm*d dST",
            "10R²K",
        ),
        "array/evens": (
            "jd:2 21 2",
            "20RḊm2K",
        ),
        "text/count_character": (
            '/"abracadabra"\\a',
            "“abracadabra”ċ”a",
        ),
        "array/dot_product": (
            "s.b*NY[1 2 3)[4 5 6)",
            "1,2,3ḋ4,5,6",
        ),
        "text/palindrome": (
            'sq"racecar"_"racecar"',
            "“racecar”Ṛ⁼",
        ),
        "scalar/gcd": (
            "i2706 410",
            "2706g410",
        ),
        "array/rotate_left": (
            "jd.<[1 2 3 4 5)3",
            "1,2,3,4,5ṙ3K",
        ),
        "recurrence/fibonacci": (
            "A(Z1)V12GA(H+HG",
            "11RŻÆḞK",
        ),
    }
    if set(competitors) != set(compact_pairs):
        missing = sorted(set(compact_pairs) - set(competitors))
        extra = sorted(set(competitors) - set(compact_pairs))
        raise RuntimeError(
            f"Golf-language registry mismatch; missing={missing}, extra={extra}"
        )
    pairs = [
        GolfPair(
            task_id=pair.task_id,
            category=pair.category,
            python=pair.python,
            pyth=competitors[pair.task_id][0] + "\n",
            jelly=competitors[pair.task_id][1] + "\n",
            expected_stdout=pair.expected_stdout,
        )
        for pair in paired_programs()
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} golf-language pairs.")
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


def _combined_output(command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _jelly_runtime_metadata(jelly_python: Path) -> dict[str, str]:
    script = (
        "import importlib.metadata,json,jelly.interpreter,pathlib,sympy;"
        "print(json.dumps({"
        "'interpreter':str(pathlib.Path(jelly.interpreter.__file__).resolve()),"
        "'package':importlib.metadata.version('jellylanguage'),"
        "'sympy':sympy.__version__,"
        "'code_page':jelly.interpreter.code_page"
        "}))"
    )
    ok, stdout, error = run_command([str(jelly_python), "-c", script])
    if not ok:
        raise RuntimeError(f"Unable to inspect Jelly runtime: {error}")
    return json.loads(stdout)


def runtime_gates(
    *,
    pyth_root: Path,
    pyth_python: Path,
    jelly_root: Path,
    jelly_python: Path,
    jelly_binary: Path,
) -> tuple[dict[str, Any], str]:
    pyth_commit = git_commit(pyth_root)
    jelly_commit = git_commit(jelly_root)
    pyth_version_ok, pyth_version = _combined_output(
        [str(pyth_python), "--version"]
    )
    jelly_version_ok, jelly_version = _combined_output(
        [str(jelly_python), "--version"]
    )
    jelly_metadata = _jelly_runtime_metadata(jelly_python)
    jelly_interpreter = Path(jelly_metadata["interpreter"])
    jelly_repo_interpreter = jelly_root / "jelly" / "interpreter.py"
    actual_jelly_hash = sha256_file(jelly_interpreter)
    repo_jelly_hash = sha256_file(jelly_repo_interpreter)
    gates = {
        "pyth": {
            "ok": (
                pyth_commit == PYTH_COMMIT
                and pyth_version_ok
                and pyth_version.endswith(PYTH_PYTHON_VERSION)
            ),
            "commit": pyth_commit,
            "expected_commit": PYTH_COMMIT,
            "python_version": pyth_version,
            "expected_python_version": PYTH_PYTHON_VERSION,
            "interpreter_sha256": sha256_file(pyth_root / "pyth.py"),
        },
        "jelly": {
            "ok": (
                jelly_commit == JELLY_COMMIT
                and jelly_version_ok
                and actual_jelly_hash == repo_jelly_hash
                and len(jelly_metadata["code_page"]) == 256
            ),
            "commit": jelly_commit,
            "expected_commit": JELLY_COMMIT,
            "python_version": jelly_version,
            "package_version": jelly_metadata["package"],
            "sympy_version": jelly_metadata["sympy"],
            "runtime_interpreter_sha256": actual_jelly_hash,
            "repository_interpreter_sha256": repo_jelly_hash,
            "binary_sha256": sha256_file(jelly_binary),
            "code_page_characters": len(jelly_metadata["code_page"]),
        },
    }
    return gates, jelly_metadata["code_page"]


def score_pairs(
    *,
    pairs: list[GolfPair],
    pyth_root: Path,
    pyth_python: Path,
    jelly_binary: Path,
    jelly_code_page: str,
    tokenizer: Tokenizer,
) -> list[GolfResult]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    results: list[GolfResult] = []
    with tempfile.TemporaryDirectory(prefix="kern-golf-language-pairs-") as raw:
        temp = Path(raw)
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
                "pyth": pair.pyth.strip(),
                "jelly": pair.jelly.strip(),
            }
            unknown_jelly = sorted(set(sources["jelly"]) - set(jelly_code_page))
            if unknown_jelly:
                raise RuntimeError(
                    f"{pair.task_id} uses characters outside Jelly code page: "
                    f"{unknown_jelly}"
                )
            native_ids = tokenizer.encode(kern_source).ids

            python_path = temp / f"{index:02d}-python.py"
            kern_path = temp / f"{index:02d}-kern.py"
            minified_path = temp / f"{index:02d}-minified.py"
            for path, value in (
                (python_path, python_source),
                (kern_path, decoded),
                (minified_path, minified),
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
            pyth_ok, pyth_stdout, pyth_error = run_command(
                [
                    str(pyth_python),
                    str(pyth_root / "pyth.py"),
                    "-c",
                    sources["pyth"],
                ]
            )
            jelly_ok, jelly_stdout, jelly_error = run_command(
                [str(jelly_binary), "eun", sources["jelly"]]
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
            pyth_matches = normalize_stdout(pyth_stdout) == expected
            jelly_matches = normalize_stdout(jelly_stdout) == expected
            results.append(
                GolfResult(
                    task_id=pair.task_id,
                    category=pair.category,
                    python_sha256=sha256_text(python_source),
                    kern_sha256=sha256_text(kern_source),
                    python_minifier_sha256=sha256_text(minified),
                    pyth_sha256=sha256_text(sources["pyth"]),
                    jelly_sha256=sha256_text(sources["jelly"]),
                    python_bytes=byte_counts["python"],
                    kern_bytes=byte_counts["kern"],
                    python_minifier_bytes=byte_counts["python_minifier"],
                    pyth_bytes=byte_counts["pyth"],
                    jelly_bytes=byte_counts["jelly"],
                    jelly_code_page_units=len(sources["jelly"]),
                    python_cl100k=cl["python"],
                    kern_cl100k=cl["kern"],
                    python_minifier_cl100k=cl["python_minifier"],
                    pyth_cl100k=cl["pyth"],
                    jelly_cl100k=cl["jelly"],
                    python_o200k=o["python"],
                    kern_o200k=o["kern"],
                    python_minifier_o200k=o["python_minifier"],
                    pyth_o200k=o["pyth"],
                    jelly_o200k=o["jelly"],
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
                    pyth_oracle_ok=pyth_ok and pyth_matches,
                    jelly_oracle_ok=jelly_ok and jelly_matches,
                    pyth_error=(
                        "" if pyth_ok and pyth_matches
                        else (pyth_error or pyth_stdout)[-1_000:]
                    ),
                    jelly_error=(
                        "" if jelly_ok and jelly_matches
                        else (jelly_error or jelly_stdout)[-1_000:]
                    ),
                )
            )
    return results


def aggregate(results: list[GolfResult]) -> dict[str, Any]:
    representations = (
        "python",
        "kern",
        "python_minifier",
        "pyth",
        "jelly",
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
        "pyth": sum(result.pyth_oracle_ok for result in results),
        "jelly": sum(result.jelly_oracle_ok for result in results),
    }
    native_kern = sum(result.kern_native_16k for result in results)
    competitors = ("pyth", "jelly")
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
        "jelly_code_page_units": sum(
            result.jelly_code_page_units for result in results
        ),
        "native_system": {
            "kern_native_16k": native_kern,
            "pyth_cl100k_base": cl["pyth"],
            "jelly_cl100k_base": cl["jelly"],
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


def write_details(results: list[GolfResult], path: Path) -> None:
    fields = list(asdict(results[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_token_svg(path: Path, row: dict[str, Any]) -> None:
    write_grouped_bar_svg(
        path,
        title="Kern versus Pyth and Jelly",
        subtitle=(
            "14 matched executable programs · complete sources · "
            "lower is better"
        ),
        groups=["cl100k_base", "o200k_base"],
        series=[
            (
                "Python",
                "#94a3b8",
                [
                    row["cl100k_base"]["python"],
                    row["o200k_base"]["python"],
                ],
            ),
            (
                "python-minifier",
                "#06b6d4",
                [
                    row["cl100k_base"]["python_minifier"],
                    row["o200k_base"]["python_minifier"],
                ],
            ),
            (
                "Pyth",
                "#a855f7",
                [
                    row["cl100k_base"]["pyth"],
                    row["o200k_base"]["pyth"],
                ],
            ),
            (
                "Jelly",
                "#f97316",
                [
                    row["cl100k_base"]["jelly"],
                    row["o200k_base"]["jelly"],
                ],
            ),
            (
                "Kern compact",
                "#22c55e",
                [
                    row["cl100k_base"]["kern"],
                    row["o200k_base"]["kern"],
                ],
            ),
        ],
        y_label="Aggregate LLM tokens",
        max_value=300.0,
        value_suffix="",
    )


def write_system_svg(path: Path, row: dict[str, Any]) -> None:
    shared = row["cl100k_base"]
    native = row["native_system"]["kern_native_16k"]
    write_grouped_bar_svg(
        path,
        title="Native-system lane crosses the golf frontier",
        subtitle=(
            "Competitor sources use cl100k_base; Kern is shown with both "
            "tokenizers"
        ),
        groups=["Pyth", "Jelly"],
        series=[
            (
                "Competitor + cl100k",
                "#f97316",
                [shared["pyth"], shared["jelly"]],
            ),
            (
                "Kern + cl100k",
                "#06b6d4",
                [shared["kern"], shared["kern"]],
            ),
            (
                "Kern + Kern-16K",
                "#22c55e",
                [native, native],
            ),
        ],
        y_label="Aggregate tokens",
        max_value=220.0,
        value_suffix="",
    )


def write_byte_svg(path: Path, row: dict[str, Any]) -> None:
    values = row["utf8_bytes"]
    write_grouped_bar_svg(
        path,
        title="Complete-source UTF-8 byte accounting",
        subtitle=(
            "Jelly code-page scoring is excluded here and reported separately"
        ),
        groups=["14 programs"],
        series=[
            ("Python", "#94a3b8", [values["python"]]),
            (
                "python-minifier",
                "#06b6d4",
                [values["python_minifier"]],
            ),
            ("Pyth", "#a855f7", [values["pyth"]]),
            ("Jelly", "#f97316", [values["jelly"]]),
            ("Kern compact", "#22c55e", [values["kern"]]),
        ],
        y_label="UTF-8 bytes",
        max_value=700.0,
        value_suffix="",
    )


def write_functional_svg(path: Path, row: dict[str, Any]) -> None:
    total = row["programs"]
    functional = row["functional"]
    write_grouped_bar_svg(
        path,
        title="Golf-language functional preservation",
        subtitle=(
            "Every complete source is executed against the same normalized "
            "stdout oracle"
        ),
        groups=["14 matched programs"],
        series=[
            ("Kern", "#22c55e", [functional["kern"] / total * 100]),
            ("Pyth", "#a855f7", [functional["pyth"] / total * 100]),
            ("Jelly", "#f97316", [functional["jelly"] / total * 100]),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyth-root", type=Path, required=True)
    parser.add_argument("--pyth-python", type=Path, required=True)
    parser.add_argument("--jelly-root", type=Path, required=True)
    parser.add_argument("--jelly-python", type=Path, required=True)
    parser.add_argument("--jelly-binary", type=Path, required=True)
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
        default=Path("benchmark_results/golf-languages"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = (
        args.pyth_root / "pyth.py",
        args.pyth_python,
        args.jelly_root / "jelly" / "interpreter.py",
        args.jelly_python,
        args.jelly_binary,
        args.tokenizer,
        args.tokenizer_manifest,
    )
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Required runtime artifact is missing: {path}")

    gates, jelly_code_page = runtime_gates(
        pyth_root=args.pyth_root,
        pyth_python=args.pyth_python,
        jelly_root=args.jelly_root,
        jelly_python=args.jelly_python,
        jelly_binary=args.jelly_binary,
    )
    failed_gates = [name for name, gate in gates.items() if not gate["ok"]]
    if failed_gates:
        raise RuntimeError(
            "Golf-language runtime gates failed: " + ", ".join(failed_gates)
        )

    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern tokenizer SHA-256 does not match manifest.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    pairs = golf_pairs()
    results = score_pairs(
        pairs=pairs,
        pyth_root=args.pyth_root,
        pyth_python=args.pyth_python,
        jelly_binary=args.jelly_binary,
        jelly_code_page=jelly_code_page,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    failed_oracles = {
        language: aggregate_row["programs"] - passed
        for language, passed in aggregate_row["functional"].items()
        if passed != aggregate_row["programs"]
    }
    if failed_oracles:
        raise RuntimeError(f"Golf-language oracle failures: {failed_oracles}")

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Fourteen fixed matched executable programs; complete public "
                "sources; shared production tokenizers; exact normalized "
                "stdout; separate native-system and Jelly code-page lanes"
            ),
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "discovery_evidence": {
            "source_snapshot_date": "2026-07-29",
            "official_repositories": {
                "pyth": "https://github.com/isaacg1/pyth",
                "jelly": "https://github.com/DennisMitchell/jellylanguage",
            },
            "interpretation": (
                "Pyth and Jelly are adversarial compact-language baselines; "
                "this fixed corpus is a density screen, not a claim about all "
                "programs or globally minimal golf solutions"
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
            "jelly_code_page_policy": (
                "Every Jelly source character must belong to the official "
                "256-character code page; code-page units are separate from "
                "UTF-8 bytes and LLM tokens"
            ),
        },
        "results": aggregate_row,
    }
    (args.output_dir / "golf-language-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "golf-language-corpus.json").write_text(
        json.dumps(
            [asdict(pair) for pair in pairs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_details(results, args.output_dir / "golf-language-details.csv")
    write_token_svg(
        args.output_dir / "golf-language-token-density.svg",
        aggregate_row,
    )
    write_system_svg(
        args.output_dir / "golf-language-native-system.svg",
        aggregate_row,
    )
    write_byte_svg(
        args.output_dir / "golf-language-utf8-bytes.svg",
        aggregate_row,
    )
    write_functional_svg(
        args.output_dir / "golf-language-functional.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
