"""Executable token-density screen for Kern and Vyxal 3.

The benchmark reuses the fixed fourteen-program compact-language corpus.  It
executes Vyxal from its official one-byte code page, verifies the pinned release
JAR, and keeps LLM tokens, UTF-8 bytes, and Vyxal code-page units separate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import platform
import shutil
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
    package_version,
    pct_below,
    run_command,
    sha256_file,
    sha256_text,
    write_grouped_bar_svg,
)
from benchmark_golf_languages import golf_pairs
from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile

VYXAL_VERSION = "3.12.0"
VYXAL_RELEASE = "v3.12.0"
VYXAL_COMMIT = "7f201806cde2a1fafdca054ac398be36f939c273"
VYXAL_JAR_SHA256 = (
    "f50af719c56374534216912887097959e8cc58dd8622491c0a246f8479cb7615"
)
VYXAL_RELEASE_URL = "https://github.com/Vyxal/Vyxal/releases/tag/v3.12.0"
EXPECTED_PAIRS = 14

# Copied from the pinned Vyxal source's vyxal.parsing.Codepage value. Literal
# newlines are formatting only; the visible ␤ position becomes code-page LF.
VYXAL_CODEPAGE = r'''λƛʎµξ⍾⎋⍟⎊⎄␤⩔Ẅ⊐⎇¿
∥∦∺⁜⑴⑵⑶⑷⎂⟒ᛞ▦¨⊞×÷
 !"#$%&'()*+,-./
0123456789:;<=>?
@ABCDEFGHIJKLMNO
PQRSTUVWXYZ[\]^_
`abcdefghijklmno
pqrstuvwxyz{|}~◲
⨥⨪∑Π⇧⇩∪∩⊍⦰«»ƓɠĠġ
⌈⌊⊖⌽£¥↜↝↺↻≜⎀⊢⊣ɦʈ
ᐐᐵᐕ½ƶƵ⁰¹²³⅟※⇄⧖‰≛
ℭ℈⦷Ϣ≤≥≠≡•±†⎙γ≓Ͼᴥ
ℳ℗↸⍢ℂ⌹⏚↯⊠⚅æ␣¶★ᑂ∻
√⍰◌δ☷σ⎶⊆⍨⎘ꜝ≈≊κ‹›
ʀʁɾ▲ṬṪ⤻⤺Ŀ¬∧∨ŁḧᏜᏐ
¤⧢①②③④⑤⑥⑦⑧Þ∆ø„”“'''.replace("\n", "").replace("␤", "\n")

if len(VYXAL_CODEPAGE) != 256 or len(set(VYXAL_CODEPAGE)) != 256:
    raise RuntimeError("The pinned Vyxal code page must contain 256 unique units")


@dataclass(frozen=True)
class VyxalPair:
    task_id: str
    category: str
    python: str
    vyxal: str
    expected_stdout: str


@dataclass
class VyxalResult:
    task_id: str
    category: str
    python_sha256: str
    kern_sha256: str
    python_minifier_sha256: str
    vyxal_sha256: str
    vyxal_codepage_sha256: str
    python_bytes: int
    kern_bytes: int
    python_minifier_bytes: int
    vyxal_bytes: int
    vyxal_codepage_units: int
    python_cl100k: int
    kern_cl100k: int
    python_minifier_cl100k: int
    vyxal_cl100k: int
    python_o200k: int
    kern_o200k: int
    python_minifier_o200k: int
    vyxal_o200k: int
    kern_native_16k: int
    kern_native_exact_roundtrip: bool
    kern_contract_ast: bool
    vyxal_codepage_roundtrip: bool
    python_oracle_ok: bool
    kern_oracle_ok: bool
    python_minifier_oracle_ok: bool
    vyxal_oracle_ok: bool
    vyxal_error: str


def vyxal_pairs() -> list[VyxalPair]:
    """Join fixed Vyxal sources to the canonical fourteen-task registry."""
    base = {pair.task_id: pair for pair in golf_pairs()}
    sources = {
        "scalar/arithmetic": "17 23×11+2∻",
        "reduction/sum_1_100": "100ɾ∑",
        "reduction/factorial_10": "10!",
        "text/reverse": '"kernlanguage"⇄',
        "array/sort": "#[9|1|5|3|7|2|8|6|4#]S„",
        "array/distinct": "#[3|1|2|3|2|4|1|5#]u„",
        "array/squares": "10ɾ²„",
        "array/evens": "20ɾʎe}„",
        "text/count_character": '"abracadabra"\'aC',
        "array/dot_product": "#[1|2|3#]#[4|5|6#]ᛞ×+",
        "text/palindrome": '"racecar"/⇄',
        "scalar/gcd": "2706 410κ",
        "array/rotate_left": "#[1|2|3|4|5#]3↺„",
        "recurrence/fibonacci": "#[0|1#]⎄+}12⊖„",
    }
    if set(sources) != set(base):
        missing = sorted(set(base) - set(sources))
        extra = sorted(set(sources) - set(base))
        raise RuntimeError(
            f"Vyxal registry mismatch; missing={missing}, extra={extra}"
        )
    pairs = [
        VyxalPair(
            task_id=pair.task_id,
            category=pair.category,
            python=pair.python,
            vyxal=sources[pair.task_id],
            expected_stdout=pair.expected_stdout,
        )
        for pair in golf_pairs()
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} Vyxal pairs")
    return pairs


def encode_vyxal_codepage(source: str) -> bytes:
    """Encode Unicode SBCS source into Vyxal's official one-byte code page."""
    missing = sorted(set(source) - set(VYXAL_CODEPAGE))
    if missing:
        raise ValueError(f"Vyxal source contains non-code-page units: {missing}")
    return bytes(VYXAL_CODEPAGE.index(char) for char in source)


def runtime_gate(jar: Path) -> dict[str, Any]:
    """Verify the exact official release artifact and local Java runtime."""
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java is required to execute Vyxal")
    version_ok, version_stdout, version_error = run_command(
        [java, "-jar", str(jar), "--version"]
    )
    java_result = subprocess.run(
        [java, "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    jar_hash = sha256_file(jar)
    return {
        "ok": (
            version_ok
            and version_stdout == VYXAL_VERSION
            and jar_hash == VYXAL_JAR_SHA256
        ),
        "version": version_stdout,
        "expected_version": VYXAL_VERSION,
        "version_error": version_error,
        "release": VYXAL_RELEASE,
        "source_commit": VYXAL_COMMIT,
        "jar_sha256": jar_hash,
        "expected_jar_sha256": VYXAL_JAR_SHA256,
        "jar_bytes": jar.stat().st_size,
        "java": (java_result.stderr or java_result.stdout).splitlines()[0],
        "codepage_units": len(VYXAL_CODEPAGE),
        "codepage_sha256": sha256_text(VYXAL_CODEPAGE),
    }


def score_pairs(
    *,
    pairs: list[VyxalPair],
    vyxal_jar: Path,
    tokenizer: Tokenizer,
) -> list[VyxalResult]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("Java is required to execute Vyxal")
    results: list[VyxalResult] = []
    with tempfile.TemporaryDirectory(prefix="kern-vyxal-pairs-") as raw:
        temp = Path(raw)
        for index, pair in enumerate(pairs):
            python_source = pair.python.strip()
            expected_compact = ast.unparse(compact_tree(ast.parse(python_source)))
            kern_source = transpile(python_source, compact=True).strip()
            decoded = compile_kern(kern_source)
            minified = python_minifier.minify(
                python_source,
                rename_globals=False,
            )
            vyxal_source = pair.vyxal.strip()
            vyxal_codepage = encode_vyxal_codepage(vyxal_source)
            sources = {
                "python": python_source,
                "kern": kern_source,
                "python_minifier": minified,
                "vyxal": vyxal_source,
            }
            native_ids = tokenizer.encode(kern_source).ids

            python_path = temp / f"{index:02d}-python.py"
            kern_path = temp / f"{index:02d}-kern.py"
            minified_path = temp / f"{index:02d}-minified.py"
            vyxal_path = temp / f"{index:02d}-vyxal.vy"
            python_path.write_text(python_source + "\n", encoding="utf-8")
            kern_path.write_text(decoded + "\n", encoding="utf-8")
            minified_path.write_text(minified + "\n", encoding="utf-8")
            vyxal_path.write_bytes(vyxal_codepage)

            python_ok, python_stdout, _ = run_command(
                [sys.executable, str(python_path)]
            )
            kern_ok, kern_stdout, _ = run_command(
                [sys.executable, str(kern_path)]
            )
            minifier_ok, minifier_stdout, _ = run_command(
                [sys.executable, str(minified_path)]
            )
            vyxal_ok, vyxal_stdout, vyxal_error = run_command(
                [
                    java,
                    "-jar",
                    str(vyxal_jar),
                    "--bytes",
                    "--file",
                    str(vyxal_path),
                ]
            )
            expected = normalize_stdout(pair.expected_stdout)
            cl = {
                name: len(encodings["cl100k_base"].encode_ordinary(value))
                for name, value in sources.items()
            }
            o = {
                name: len(encodings["o200k_base"].encode_ordinary(value))
                for name, value in sources.items()
            }
            byte_counts = {
                name: len(value.encode("utf-8"))
                for name, value in sources.items()
            }
            vyxal_matches = normalize_stdout(vyxal_stdout) == expected
            results.append(
                VyxalResult(
                    task_id=pair.task_id,
                    category=pair.category,
                    python_sha256=sha256_text(python_source),
                    kern_sha256=sha256_text(kern_source),
                    python_minifier_sha256=sha256_text(minified),
                    vyxal_sha256=sha256_text(vyxal_source),
                    vyxal_codepage_sha256=hashlib.sha256(
                        vyxal_codepage
                    ).hexdigest(),
                    python_bytes=byte_counts["python"],
                    kern_bytes=byte_counts["kern"],
                    python_minifier_bytes=byte_counts["python_minifier"],
                    vyxal_bytes=byte_counts["vyxal"],
                    vyxal_codepage_units=len(vyxal_codepage),
                    python_cl100k=cl["python"],
                    kern_cl100k=cl["kern"],
                    python_minifier_cl100k=cl["python_minifier"],
                    vyxal_cl100k=cl["vyxal"],
                    python_o200k=o["python"],
                    kern_o200k=o["kern"],
                    python_minifier_o200k=o["python_minifier"],
                    vyxal_o200k=o["vyxal"],
                    kern_native_16k=len(native_ids),
                    kern_native_exact_roundtrip=(
                        tokenizer.decode(native_ids) == kern_source
                    ),
                    kern_contract_ast=(
                        normalize_ast(decoded) == normalize_ast(expected_compact)
                    ),
                    vyxal_codepage_roundtrip=(
                        "".join(VYXAL_CODEPAGE[value] for value in vyxal_codepage)
                        == vyxal_source
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
                    vyxal_oracle_ok=vyxal_ok and vyxal_matches,
                    vyxal_error=(
                        "" if vyxal_ok and vyxal_matches
                        else (vyxal_error or vyxal_stdout)[-1_000:]
                    ),
                )
            )
    return results


def aggregate(results: list[VyxalResult]) -> dict[str, Any]:
    representations = ("python", "kern", "python_minifier", "vyxal")
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
        "vyxal": sum(result.vyxal_oracle_ok for result in results),
    }
    native_kern = sum(result.kern_native_16k for result in results)
    categories: dict[str, Any] = {}
    for category in sorted({result.category for result in results}):
        selected = [
            result for result in results if result.category == category
        ]
        categories[category] = {
            "programs": len(selected),
            "cl100k_base": {
                name: sum(
                    getattr(result, f"{name}_cl100k") for result in selected
                )
                for name in representations
            },
            "vyxal_codepage_units": sum(
                result.vyxal_codepage_units for result in selected
            ),
        }
    return {
        "programs": len(results),
        "cl100k_base": cl,
        "o200k_base": o,
        "utf8_bytes": byte_totals,
        "vyxal_codepage_units": sum(
            result.vyxal_codepage_units for result in results
        ),
        "native_system": {
            "kern_native_16k": native_kern,
            "vyxal_cl100k_base": cl["vyxal"],
        },
        "functional": functional,
        "structural": {
            "kern_contract_ast": sum(
                result.kern_contract_ast for result in results
            ),
            "kern_native_exact_roundtrip": sum(
                result.kern_native_exact_roundtrip for result in results
            ),
            "vyxal_codepage_roundtrip": sum(
                result.vyxal_codepage_roundtrip for result in results
            ),
        },
        "comparisons": {
            "shared_kern_below_vyxal_pct": pct_below(cl["kern"], cl["vyxal"]),
            "o200k_kern_below_vyxal_pct": pct_below(o["kern"], o["vyxal"]),
            "native_kern_below_vyxal_cl100k_pct": pct_below(
                native_kern, cl["vyxal"]
            ),
            "shared_kern_wins": sum(
                result.kern_cl100k < result.vyxal_cl100k
                for result in results
            ),
            "shared_ties": sum(
                result.kern_cl100k == result.vyxal_cl100k
                for result in results
            ),
            "shared_vyxal_wins": sum(
                result.vyxal_cl100k < result.kern_cl100k
                for result in results
            ),
        },
        "categories": categories,
    }


def write_details(results: list[VyxalResult], path: Path) -> None:
    fields = list(asdict(results[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_token_svg(path: Path, row: dict[str, Any]) -> None:
    write_grouped_bar_svg(
        path,
        title="Kern versus Vyxal 3.12.0",
        subtitle="14 executable programs · complete Unicode sources",
        groups=["cl100k_base", "o200k_base"],
        series=[
            (
                "Python",
                "#94a3b8",
                [row["cl100k_base"]["python"], row["o200k_base"]["python"]],
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
                "Vyxal",
                "#f97316",
                [row["cl100k_base"]["vyxal"], row["o200k_base"]["vyxal"]],
            ),
            (
                "Kern compact",
                "#22c55e",
                [row["cl100k_base"]["kern"], row["o200k_base"]["kern"]],
            ),
        ],
        y_label="Aggregate LLM tokens",
        max_value=300.0,
        value_suffix="",
    )


def write_system_svg(path: Path, row: dict[str, Any]) -> None:
    write_grouped_bar_svg(
        path,
        title="Deployable tokenizer lane",
        subtitle="Vyxal uses cl100k; Kern is shown with shared and native BPE",
        groups=["14 programs"],
        series=[
            ("Vyxal + cl100k", "#f97316", [row["cl100k_base"]["vyxal"]]),
            ("Kern + cl100k", "#06b6d4", [row["cl100k_base"]["kern"]]),
            (
                "Kern + Kern-16K",
                "#22c55e",
                [row["native_system"]["kern_native_16k"]],
            ),
        ],
        y_label="Aggregate tokens",
        max_value=170.0,
        value_suffix="",
    )


def write_source_units_svg(path: Path, row: dict[str, Any]) -> None:
    write_grouped_bar_svg(
        path,
        title="Source storage units are not LLM tokens",
        subtitle="UTF-8 bytes and Vyxal code-page units are reported separately",
        groups=["UTF-8 bytes", "Vyxal code page"],
        series=[
            ("Kern compact", "#22c55e", [row["utf8_bytes"]["kern"], None]),
            (
                "Vyxal",
                "#f97316",
                [row["utf8_bytes"]["vyxal"], row["vyxal_codepage_units"]],
            ),
        ],
        y_label="Aggregate source units",
        max_value=240.0,
        value_suffix="",
    )


def write_functional_svg(path: Path, row: dict[str, Any]) -> None:
    total = row["programs"]
    functional = row["functional"]
    write_grouped_bar_svg(
        path,
        title="Vyxal functional gate",
        subtitle="Official code-page bytes executed against exact stdout oracles",
        groups=["14 matched programs"],
        series=[
            ("Kern", "#22c55e", [functional["kern"] / total * 100]),
            ("Vyxal", "#f97316", [functional["vyxal"] / total * 100]),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vyxal-jar",
        type=Path,
        default=Path("external/Vyxal/vyxal-3.12.0.jar"),
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
        default=Path("benchmark_results/vyxal"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        args.vyxal_jar,
        args.tokenizer,
        args.tokenizer_manifest,
    ):
        if not required.exists():
            raise RuntimeError(f"Required artifact is missing: {required}")

    gate = runtime_gate(args.vyxal_jar)
    if not gate["ok"]:
        raise RuntimeError(f"Vyxal runtime gate failed: {gate}")
    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern tokenizer SHA-256 does not match manifest")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    pairs = vyxal_pairs()
    results = score_pairs(
        pairs=pairs,
        vyxal_jar=args.vyxal_jar,
        tokenizer=tokenizer,
    )
    aggregate_row = aggregate(results)
    failed_oracles = {
        language: aggregate_row["programs"] - passed
        for language, passed in aggregate_row["functional"].items()
        if passed != aggregate_row["programs"]
    }
    if failed_oracles:
        raise RuntimeError(f"Vyxal benchmark oracle failures: {failed_oracles}")
    failed_structural = {
        name: aggregate_row["programs"] - passed
        for name, passed in aggregate_row["structural"].items()
        if passed != aggregate_row["programs"]
    }
    if failed_structural:
        raise RuntimeError(f"Vyxal structural failures: {failed_structural}")

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Fourteen fixed matched executable programs; official Vyxal "
                "code-page execution; shared production tokenizers; exact "
                "normalized stdout; separate LLM-token, native-BPE, UTF-8, "
                "and Vyxal code-page lanes"
            ),
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "discovery_evidence": {
            "source_snapshot_date": "2026-08-01",
            "official_repository": "https://github.com/Vyxal/Vyxal",
            "official_release": VYXAL_RELEASE_URL,
            "interpretation": (
                "Vyxal is an adversarial modern golfing language. This fixed "
                "corpus is a density screen, not a claim that either source "
                "set is globally minimal."
            ),
        },
        "runtime_gate": gate,
        "corpus": {
            "authorship": (
                "Benchmark-authored compact Vyxal programs using documented "
                "primitives; not certified best-known golf submissions"
            ),
            "normalization": (
                "Collapse display-only whitespace while preserving value "
                "tokens and their order"
            ),
            "sources_and_hashes_published": True,
        },
        "results": aggregate_row,
    }
    (args.output_dir / "vyxal-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "vyxal-corpus.json").write_text(
        json.dumps(
            [asdict(pair) for pair in pairs],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_details(results, args.output_dir / "vyxal-details.csv")
    write_token_svg(args.output_dir / "vyxal-token-density.svg", aggregate_row)
    write_system_svg(args.output_dir / "vyxal-system-lane.svg", aggregate_row)
    write_source_units_svg(
        args.output_dir / "vyxal-source-units.svg",
        aggregate_row,
    )
    write_functional_svg(
        args.output_dir / "vyxal-functional.svg",
        aggregate_row,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
