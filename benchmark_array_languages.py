"""Executable token-density screen for Kern, Uiua, and BQN.

Uiua and BQN are modern array-language baselines rather than Python source
minifiers. This harness uses a small fixed corpus of matched, inspectable
programs, executes every representation against the same stdout oracle, and
counts every complete source under the same production tokenizers.
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

UIUA_VERSION = "uiua 0.18.1"
UIUA_BINARY_SHA256 = (
    "d4585363ebac31c6d63575108c13aff796fe86050f2b975ccff4f8bde22fd114"
)
CBQN_COMMIT = "b4db324a99d6590d91b9b09bc36847f3254c1543"
CBQN_TAG = "v0.12.0"
CBQN_BINARY_SHA256 = (
    "32c0915af389cc469cb3f663025d72c7aab39ca451d02de41b4c24f7b8e338e6"
)
EXPECTED_PAIRS = 14


@dataclass(frozen=True)
class ArrayPair:
    task_id: str
    category: str
    python: str
    uiua: str
    bqn: str
    bqn_mode: str
    expected_stdout: str


@dataclass
class ArrayResult:
    task_id: str
    category: str
    python_sha256: str
    kern_sha256: str
    python_minifier_sha256: str
    uiua_sha256: str
    bqn_sha256: str
    python_bytes: int
    kern_bytes: int
    python_minifier_bytes: int
    uiua_bytes: int
    bqn_bytes: int
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    uiua_cl100k: int
    bqn_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    uiua_o200k: int
    bqn_o200k: int
    kern_native_16k: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    uiua_oracle_ok: bool
    bqn_oracle_ok: bool
    uiua_error: str
    bqn_error: str


def array_pairs() -> list[ArrayPair]:
    """Return the fixed matched-program registry."""
    compact_pairs = {pair.task_id: pair for pair in paired_programs()}
    competitors = {
        "scalar/arithmetic": (
            "&p÷2+11×17 23",
            "⌊2÷˜11+17×23",
            "p",
        ),
        "reduction/sum_1_100": (
            "&p/++1⇡100",
            "+´1+↕100",
            "p",
        ),
        "reduction/factorial_10": (
            "&p /× +1 ⇡10",
            "×´1+↕10",
            "p",
        ),
        "text/reverse": (
            '&p ⇌ "kernlanguage"',
            '⌽"kernlanguage"',
            "o",
        ),
        "array/sort": (
            "≡&p⍆9_1_5_3_7_2_8_6_4",
            "•Out¨•Fmt¨∧9‿1‿5‿3‿7‿2‿8‿6‿4",
            "e",
        ),
        "array/distinct": (
            "≡&p◴3_1_2_3_2_4_1_5",
            "•Out¨•Fmt¨⍷3‿1‿2‿3‿2‿4‿1‿5",
            "e",
        ),
        "array/squares": (
            "≡&p ×⊸∘ +1 ⇡10",
            "•Out¨•Fmt¨×˜1+↕10",
            "e",
        ),
        "array/evens": (
            "≡&p ×2 +1 ⇡10",
            "•Out¨•Fmt¨2×1+↕10",
            "e",
        ),
        "text/count_character": (
            '&p/+=@a"abracadabra"',
            '+´\'a\'="abracadabra"',
            "p",
        ),
        "array/dot_product": (
            "&p/+×1_2_3 4_5_6",
            "+´1‿2‿3×4‿5‿6",
            "p",
        ),
        "text/palindrome": (
            '&p≍⇌⊸∘"racecar"',
            '"racecar"≡⌽"racecar"',
            "p",
        ),
        "scalar/gcd": (
            "&p ⊢⍢([⊃⊢(/◿)↻1]|≠0⊣)2706_410",
            "2706•math.GCD 410",
            "p",
        ),
        "array/rotate_left": (
            "≡&p↻3 1_2_3_4_5",
            "•Out¨•Fmt¨3⌽1‿2‿3‿4‿5",
            "e",
        ),
        "recurrence/fibonacci": (
            "≡&p⍥◡+12 1 0",
            "•Out¨•Fmt¨{𝕩∾+´¯2↑𝕩}⍟10⊢0‿1",
            "e",
        ),
    }
    if set(competitors) != set(compact_pairs):
        missing = sorted(set(compact_pairs) - set(competitors))
        extra = sorted(set(competitors) - set(compact_pairs))
        raise RuntimeError(
            f"Array-language registry mismatch; missing={missing}, extra={extra}"
        )
    pairs = [
        ArrayPair(
            task_id=pair.task_id,
            category=pair.category,
            python=pair.python,
            uiua=competitors[pair.task_id][0] + "\n",
            bqn=competitors[pair.task_id][1] + "\n",
            bqn_mode=competitors[pair.task_id][2],
            expected_stdout=pair.expected_stdout,
        )
        for pair in paired_programs()
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} array-language pairs.")
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


def runtime_gates(
    *,
    uiua_binary: Path,
    cbqn_root: Path,
    bqn_binary: Path,
) -> dict[str, Any]:
    bqn_commit = git_commit(cbqn_root)
    tag_ok, bqn_tag = _combined_output(
        ["git", "-C", str(cbqn_root), "describe", "--tags", "--exact-match"]
    )
    uiua_version_ok, uiua_version = _combined_output(
        [str(uiua_binary), "--version"]
    )
    bqn_version_ok, bqn_version = _combined_output(
        [str(bqn_binary), "--version"]
    )
    uiua_hash = sha256_file(uiua_binary)
    bqn_hash = sha256_file(bqn_binary)
    gates = {
        "uiua": {
            "ok": (
                uiua_version_ok
                and uiua_version == UIUA_VERSION
                and uiua_hash == UIUA_BINARY_SHA256
            ),
            "version": uiua_version,
            "expected_version": UIUA_VERSION,
            "binary_sha256": uiua_hash,
            "expected_binary_sha256": UIUA_BINARY_SHA256,
        },
        "bqn": {
            "ok": (
                bqn_commit == CBQN_COMMIT
                and tag_ok
                and bqn_tag == CBQN_TAG
                and bqn_version_ok
                and CBQN_COMMIT in bqn_version
                and bqn_hash == CBQN_BINARY_SHA256
            ),
            "commit": bqn_commit,
            "expected_commit": CBQN_COMMIT,
            "tag": bqn_tag,
            "expected_tag": CBQN_TAG,
            "version": bqn_version,
            "binary_sha256": bqn_hash,
            "expected_binary_sha256": CBQN_BINARY_SHA256,
        },
    }
    return gates


def score_pairs(
    *,
    pairs: list[ArrayPair],
    uiua_binary: Path,
    bqn_binary: Path,
    tokenizer: Tokenizer,
) -> list[ArrayResult]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    results: list[ArrayResult] = []
    with tempfile.TemporaryDirectory(prefix="kern-array-language-pairs-") as raw:
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
                "uiua": pair.uiua.strip(),
                "bqn": pair.bqn.strip(),
            }
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
            uiua_ok, uiua_stdout, uiua_error = run_command(
                [str(uiua_binary), "eval", sources["uiua"]]
            )
            bqn_ok, bqn_stdout, bqn_error = run_command(
                [str(bqn_binary), f"-{pair.bqn_mode}", sources["bqn"]]
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
            uiua_matches = normalize_stdout(uiua_stdout) == expected
            bqn_matches = normalize_stdout(bqn_stdout) == expected
            results.append(
                ArrayResult(
                    task_id=pair.task_id,
                    category=pair.category,
                    python_sha256=sha256_text(python_source),
                    kern_sha256=sha256_text(kern_source),
                    python_minifier_sha256=sha256_text(minified),
                    uiua_sha256=sha256_text(sources["uiua"]),
                    bqn_sha256=sha256_text(sources["bqn"]),
                    python_bytes=byte_counts["python"],
                    kern_bytes=byte_counts["kern"],
                    python_minifier_bytes=byte_counts["python_minifier"],
                    uiua_bytes=byte_counts["uiua"],
                    bqn_bytes=byte_counts["bqn"],
                    python_cl100k=cl["python"],
                    kern_cl100k=cl["kern"],
                    python_minifier_cl100k=cl["python_minifier"],
                    uiua_cl100k=cl["uiua"],
                    bqn_cl100k=cl["bqn"],
                    python_o200k=o["python"],
                    kern_o200k=o["kern"],
                    python_minifier_o200k=o["python_minifier"],
                    uiua_o200k=o["uiua"],
                    bqn_o200k=o["bqn"],
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
                    uiua_oracle_ok=uiua_ok and uiua_matches,
                    bqn_oracle_ok=bqn_ok and bqn_matches,
                    uiua_error=(
                        "" if uiua_ok and uiua_matches
                        else (uiua_error or uiua_stdout)[-1_000:]
                    ),
                    bqn_error=(
                        "" if bqn_ok and bqn_matches
                        else (bqn_error or bqn_stdout)[-1_000:]
                    ),
                )
            )
    return results


def aggregate(results: list[ArrayResult]) -> dict[str, Any]:
    representations = (
        "python",
        "kern",
        "python_minifier",
        "uiua",
        "bqn",
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
        "uiua": sum(result.uiua_oracle_ok for result in results),
        "bqn": sum(result.bqn_oracle_ok for result in results),
    }
    native_kern = sum(result.kern_native_16k for result in results)
    competitors = ("uiua", "bqn")
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
            "uiua_cl100k_base": cl["uiua"],
            "bqn_cl100k_base": cl["bqn"],
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


def write_details(results: list[ArrayResult], path: Path) -> None:
    fields = list(asdict(results[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_token_svg(path: Path, row: dict[str, Any]) -> None:
    write_grouped_bar_svg(
        path,
        title="Kern versus Uiua and BQN",
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
                "Uiua",
                "#a855f7",
                [
                    row["cl100k_base"]["uiua"],
                    row["o200k_base"]["uiua"],
                ],
            ),
            (
                "BQN",
                "#f97316",
                [
                    row["cl100k_base"]["bqn"],
                    row["o200k_base"]["bqn"],
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
        title="Native-system lane crosses the array frontier",
        subtitle=(
            "Competitor sources use cl100k_base; Kern is shown with both "
            "tokenizers"
        ),
        groups=["Uiua", "BQN"],
        series=[
            (
                "Competitor + cl100k",
                "#f97316",
                [shared["uiua"], shared["bqn"]],
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
        max_value=260.0,
        value_suffix="",
    )


def write_byte_svg(path: Path, row: dict[str, Any]) -> None:
    values = row["utf8_bytes"]
    write_grouped_bar_svg(
        path,
        title="Complete-source UTF-8 byte accounting",
        subtitle="Every glyph is counted by its encoded UTF-8 byte length",
        groups=["14 programs"],
        series=[
            ("Python", "#94a3b8", [values["python"]]),
            (
                "python-minifier",
                "#06b6d4",
                [values["python_minifier"]],
            ),
            ("Uiua", "#a855f7", [values["uiua"]]),
            ("BQN", "#f97316", [values["bqn"]]),
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
        title="Array-language functional preservation",
        subtitle=(
            "Every complete source is executed against the same normalized "
            "stdout oracle"
        ),
        groups=["14 matched programs"],
        series=[
            ("Kern", "#22c55e", [functional["kern"] / total * 100]),
            ("Uiua", "#a855f7", [functional["uiua"] / total * 100]),
            ("BQN", "#f97316", [functional["bqn"] / total * 100]),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uiua-binary", type=Path, required=True)
    parser.add_argument("--cbqn-root", type=Path, required=True)
    parser.add_argument("--bqn-binary", type=Path, required=True)
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
        default=Path("benchmark_results/array-languages"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required = (
        args.uiua_binary,
        args.cbqn_root / ".git",
        args.bqn_binary,
        args.tokenizer,
        args.tokenizer_manifest,
    )
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Required runtime artifact is missing: {path}")

    gates = runtime_gates(
        uiua_binary=args.uiua_binary,
        cbqn_root=args.cbqn_root,
        bqn_binary=args.bqn_binary,
    )
    failed_gates = [name for name, gate in gates.items() if not gate["ok"]]
    if failed_gates:
        raise RuntimeError(
            "Array-language runtime gates failed: " + ", ".join(failed_gates)
        )

    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern tokenizer SHA-256 does not match manifest.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    pairs = array_pairs()
    results = score_pairs(
        pairs=pairs,
        uiua_binary=args.uiua_binary,
        bqn_binary=args.bqn_binary,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    failed_oracles = {
        language: aggregate_row["programs"] - passed
        for language, passed in aggregate_row["functional"].items()
        if passed != aggregate_row["programs"]
    }
    if failed_oracles:
        raise RuntimeError(f"Array-language oracle failures: {failed_oracles}")

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Fourteen fixed matched executable programs; complete public "
                "sources; shared production tokenizers; exact normalized "
                "stdout; separate shared-tokenizer, native-system, and UTF-8 "
                "byte lanes"
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
                "uiua": "https://github.com/uiua-lang/uiua",
                "bqn": "https://mlochbaum.github.io/BQN/",
                "cbqn": "https://github.com/dzaima/CBQN",
            },
            "interpretation": (
                "Uiua and BQN are adversarial compact-language baselines; "
                "this fixed corpus is a density screen, not a claim about all "
                "programs or globally minimal array solutions"
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
    (args.output_dir / "array-language-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "array-language-corpus.json").write_text(
        json.dumps(
            [asdict(pair) for pair in pairs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_details(results, args.output_dir / "array-language-details.csv")
    write_token_svg(
        args.output_dir / "array-language-token-density.svg",
        aggregate_row,
    )
    write_system_svg(
        args.output_dir / "array-language-native-system.svg",
        aggregate_row,
    )
    write_byte_svg(
        args.output_dir / "array-language-utf8-bytes.svg",
        aggregate_row,
    )
    write_functional_svg(
        args.output_dir / "array-language-functional.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
