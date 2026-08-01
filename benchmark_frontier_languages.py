"""Executable frontier screen for Kern, GNU APL, CJam, and Kona K3.

The corpus is the same fourteen-program registry used by the earlier K/J,
Pyth/Jelly, and Uiua/BQN screens. Every complete source is executed against
the same normalized stdout oracle. Shared production-tokenizer scores, UTF-8
bytes, and Kern's separately labelled native-tokenizer lane are all retained.
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

EXPECTED_PAIRS = 14

GNU_APL_VERSION = "2.0"
GNU_APL_SOURCE_SHA256 = (
    "24bbb744fce47e62837234a053bdeecee51b9ea61c82c79f7cc191bc6a54c0a1"
)
KONA_COMMIT = "ac4e4c515faf586520454c266619ce1fea650554"
CJAM_VERSION = "0.6.5"
CJAM_CHANGESET = "c62f1221dfadd63f3b21776714f62573df24dd32"
CJAM_JAR_SHA256 = (
    "e7444a9ac3cab491053df2bd625217906ba07ab091ace9aa52e54f700db9e3a7"
)

REPRESENTATIONS = (
    "python",
    "kern",
    "python_minifier",
    "gnu_apl",
    "cjam",
    "kona",
)
COMPETITORS = ("gnu_apl", "cjam", "kona")
REPOSITORY_ROOT = Path(__file__).resolve().parent
PREVIOUS_EXECUTABLE_SCREENS = {
    "compact-languages": {
        "summary": REPOSITORY_ROOT
        / "benchmark_results/compact-languages/compact-language-summary.json",
        "corpus": REPOSITORY_ROOT
        / "benchmark_results/compact-languages/compact-language-corpus.json",
        "competitors": ("k", "golfscript", "j"),
    },
    "golf-languages": {
        "summary": REPOSITORY_ROOT
        / "benchmark_results/golf-languages/golf-language-summary.json",
        "corpus": REPOSITORY_ROOT
        / "benchmark_results/golf-languages/golf-language-corpus.json",
        "competitors": ("pyth", "jelly"),
    },
    "array-languages": {
        "summary": REPOSITORY_ROOT
        / "benchmark_results/array-languages/array-language-summary.json",
        "corpus": REPOSITORY_ROOT
        / "benchmark_results/array-languages/array-language-corpus.json",
        "competitors": ("uiua", "bqn"),
    },
}


@dataclass(frozen=True)
class FrontierPair:
    task_id: str
    category: str
    python: str
    gnu_apl: str
    cjam: str
    kona: str
    expected_stdout: str


def frontier_pairs() -> list[FrontierPair]:
    """Return the fixed, adversarially strengthened frontier corpus."""
    compact_pairs = {pair.task_id: pair for pair in paired_programs()}
    competitors = {
        "scalar/arithmetic": (
            "⌊.5×11+17×23",
            "23H*B+Y/",
            "(11+17*23)%2",
        ),
        "reduction/sum_1_100": (
            "+/⍳100",
            "101,:+",
            "+/1+!100",
        ),
        "reduction/factorial_10": (
            "!10",
            "Am!",
            "*/1+!10",
        ),
        "text/reverse": (
            "⌽'kernlanguage'",
            '"kernlanguage"W%',
            '`0:|"kernlanguage"',
        ),
        "array/sort": (
            "{⍵[⍋⍵]}⍎¨'915372864'",
            '"915372864"$S*',
            "{x@<x}10 _vs 915372864",
        ),
        "array/distinct": (
            "∪⍎¨'31232415'",
            '"31232415"L|S*',
            "?10 _vs 31232415",
        ),
        "array/squares": (
            "(⍳10)*2",
            "A,:)_.*S*",
            "{x*x}1+!10",
        ),
        "array/evens": (
            "2×⍳10",
            "A,:)Yf*S*",
            "2*1+!10",
        ),
        "text/count_character": (
            "+/'a'='abracadabra'",
            '"abracadabra"\'ae=',
            '+/"abracadabra"="a"',
        ),
        "array/dot_product": (
            "1 2 3+.×4 5 6",
            "123Ab456Ab.*:+",
            "+/1 2 3*4+!3",
        ),
        "text/palindrome": (
            "'racecar'≡⌽'racecar'",
            '"racecar"_W%=',
            '{x~|x}"racecar"',
        ),
        "scalar/gcd": (
            "2706∨410",
            "2706 410{_@\\%}h;",
            "a:2706!410;a!410!a",
        ),
        "array/rotate_left": (
            "3⌽⍳5",
            '"12345"3m<S*',
            "3!1+!5",
        ),
        "recurrence/fibonacci": (
            "{⍵,+/¯2↑⍵}⍣10⊢0 1",
            "T1{_2$+}A*]S*",
            "10{x,+/-2#x}/!2",
        ),
    }
    if set(competitors) != set(compact_pairs):
        missing = sorted(set(compact_pairs) - set(competitors))
        extra = sorted(set(competitors) - set(compact_pairs))
        raise RuntimeError(
            f"Frontier registry mismatch; missing={missing}, extra={extra}"
        )
    pairs = [
        FrontierPair(
            task_id=pair.task_id,
            category=pair.category,
            python=pair.python,
            gnu_apl=competitors[pair.task_id][0] + "\n",
            cjam=competitors[pair.task_id][1] + "\n",
            kona=competitors[pair.task_id][2] + "\n",
            expected_stdout=pair.expected_stdout,
        )
        for pair in paired_programs()
    ]
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} frontier pairs.")
    return pairs


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def combined_output(command: list[str]) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def run_command_in(
    command: list[str],
    *,
    cwd: Path,
) -> tuple[bool, str, str]:
    """Execute a runtime in an isolated directory and capture its output."""
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def runtime_gates(
    *,
    gnu_apl_binary: Path,
    gnu_apl_source_archive: Path,
    cjam_jar: Path,
    java: str,
    kona_root: Path,
    kona_binary: Path,
) -> dict[str, Any]:
    apl_ok, apl_version = combined_output([str(gnu_apl_binary), "--version"])
    java_ok, java_version = combined_output([java, "-version"])
    kona_commit = git_commit(kona_root)
    kona_suite_ok, kona_suite = combined_output([str(kona_root / "k_test")])
    gates = {
        "gnu_apl": {
            "ok": (
                apl_ok
                and f"Version / SVN:  {GNU_APL_VERSION}" in apl_version
                and sha256_file(gnu_apl_source_archive)
                == GNU_APL_SOURCE_SHA256
            ),
            "version": apl_version,
            "expected_version": GNU_APL_VERSION,
            "source_archive_sha256": sha256_file(
                gnu_apl_source_archive
            ),
            "expected_source_archive_sha256": GNU_APL_SOURCE_SHA256,
            "binary_sha256_observed": sha256_file(gnu_apl_binary),
        },
        "cjam": {
            "ok": (
                java_ok
                and sha256_file(cjam_jar) == CJAM_JAR_SHA256
            ),
            "version": CJAM_VERSION,
            "source_changeset": CJAM_CHANGESET,
            "jar_sha256": sha256_file(cjam_jar),
            "expected_jar_sha256": CJAM_JAR_SHA256,
            "java_version": java_version,
        },
        "kona": {
            "ok": (
                kona_commit == KONA_COMMIT
                and kona_suite_ok
                and "Failed: 0" in kona_suite
            ),
            "commit": kona_commit,
            "expected_commit": KONA_COMMIT,
            "binary_sha256_observed": sha256_file(kona_binary),
            "suite": kona_suite,
        },
    }
    failures = [name for name, gate in gates.items() if not gate["ok"]]
    if failures:
        raise RuntimeError(f"Frontier runtime gates failed: {failures}")
    return gates


def score_pairs(
    *,
    pairs: list[FrontierPair],
    gnu_apl_binary: Path,
    cjam_jar: Path,
    java: str,
    kona_binary: Path,
    tokenizer: Tokenizer,
) -> list[dict[str, Any]]:
    encodings = {
        name: tiktoken.get_encoding(name)
        for name in ("cl100k_base", "o200k_base")
    }
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="kern-frontier-language-pairs-"
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
                "gnu_apl": pair.gnu_apl.strip(),
                "cjam": pair.cjam.strip(),
                "kona": pair.kona.strip(),
            }
            native_ids = tokenizer.encode(kern_source).ids
            paths = {
                "python": temp / f"{index:02d}-python.py",
                "kern": temp / f"{index:02d}-kern.py",
                "python_minifier": temp / f"{index:02d}-minified.py",
                "cjam": temp / f"{index:02d}.cjam",
                "kona": temp / f"{index:02d}.k",
            }
            for name, path in paths.items():
                value = decoded if name == "kern" else sources[name]
                path.write_text(value + "\n", encoding="utf-8")

            executions = {
                "python": run_command([sys.executable, str(paths["python"])]),
                "kern": run_command([sys.executable, str(paths["kern"])]),
                "python_minifier": run_command(
                    [sys.executable, str(paths["python_minifier"])]
                ),
                "gnu_apl": run_command_in(
                    [
                        str(gnu_apl_binary),
                        "-s",
                        "--noSV",
                        "--eval",
                        sources["gnu_apl"],
                        "--OFF",
                    ],
                    cwd=temp,
                ),
                "cjam": run_command(
                    [java, "-jar", str(cjam_jar), str(paths["cjam"])]
                ),
                "kona": run_command(
                    [str(kona_binary), str(paths["kona"])]
                ),
            }
            expected = normalize_stdout(pair.expected_stdout)
            row: dict[str, Any] = {
                "task_id": pair.task_id,
                "category": pair.category,
                "expected_stdout": pair.expected_stdout,
                "kern_native_16k": len(native_ids),
                "kern_native_exact_roundtrip": (
                    tokenizer.decode(native_ids) == kern_source
                ),
                "kern_contract_ast": (
                    normalize_ast(decoded) == normalize_ast(expected_compact)
                ),
            }
            for name, value in sources.items():
                row[f"{name}_sha256"] = sha256_text(value)
                row[f"{name}_bytes"] = len(value.encode("utf-8"))
                row[f"{name}_cl100k"] = len(
                    encodings["cl100k_base"].encode(value)
                )
                row[f"{name}_o200k"] = len(
                    encodings["o200k_base"].encode(value)
                )
                ok, stdout, error = executions[name]
                row[f"{name}_oracle_ok"] = (
                    ok and normalize_stdout(stdout) == expected
                )
                row[f"{name}_error"] = error
            rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        lane: {
            name: sum(row[f"{name}_{suffix}"] for row in rows)
            for name in REPRESENTATIONS
        }
        for lane, suffix in (
            ("cl100k_base", "cl100k"),
            ("o200k_base", "o200k"),
            ("utf8_bytes", "bytes"),
        )
    }
    native_kern = sum(row["kern_native_16k"] for row in rows)
    categories: dict[str, Any] = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "programs": len(subset),
            "cl100k_base": {
                name: sum(row[f"{name}_cl100k"] for row in subset)
                for name in REPRESENTATIONS
            },
            "kern_native_16k": sum(
                row["kern_native_16k"] for row in subset
            ),
        }
    return {
        "programs": len(rows),
        **totals,
        "native_system": {
            "kern_native_16k": native_kern,
            **{
                f"{name}_cl100k_base": totals["cl100k_base"][name]
                for name in COMPETITORS
            },
        },
        "functional": {
            name: sum(row[f"{name}_oracle_ok"] for row in rows)
            for name in REPRESENTATIONS
        },
        "structural": {
            "kern_contract_ast": sum(
                row["kern_contract_ast"] for row in rows
            ),
            "kern_native_exact_roundtrip": sum(
                row["kern_native_exact_roundtrip"] for row in rows
            ),
        },
        "comparisons": {
            "shared_kern_below_pct": {
                name: pct_below(
                    totals["cl100k_base"]["kern"],
                    totals["cl100k_base"][name],
                )
                for name in COMPETITORS
            },
            "native_kern_below_competitor_cl100k_pct": {
                name: pct_below(
                    native_kern,
                    totals["cl100k_base"][name],
                )
                for name in COMPETITORS
            },
            "shared_kern_wins": {
                name: sum(
                    row["kern_cl100k"] < row[f"{name}_cl100k"]
                    for row in rows
                )
                for name in COMPETITORS
            },
            "native_kern_wins": {
                name: sum(
                    row["kern_native_16k"] < row[f"{name}_cl100k"]
                    for row in rows
                )
                for name in COMPETITORS
            },
            "median_per_pair_shared_kern_below_pct": {
                name: statistics.median(
                    pct_below(
                        row["kern_cl100k"],
                        row[f"{name}_cl100k"],
                    )
                    for row in rows
                )
                for name in COMPETITORS
            },
        },
        "categories": categories,
    }


def cross_screen_market(
    pairs: list[FrontierPair],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Join prior pinned screens only after proving an identical registry."""
    expected = [
        {
            "task_id": pair.task_id,
            "python": pair.python.strip(),
            "expected_stdout": pair.expected_stdout.strip(),
        }
        for pair in pairs
    ]
    competitors = {
        "cjam": current["cl100k_base"]["cjam"],
        "kona": current["cl100k_base"]["kona"],
        "gnu_apl": current["cl100k_base"]["gnu_apl"],
    }
    evidence: dict[str, Any] = {}
    for name, screen in PREVIOUS_EXECUTABLE_SCREENS.items():
        corpus = json.loads(screen["corpus"].read_text(encoding="utf-8"))
        observed = [
            {
                "task_id": row["task_id"],
                "python": row["python"].strip(),
                "expected_stdout": row["expected_stdout"].strip(),
            }
            for row in corpus
        ]
        if observed != expected:
            raise RuntimeError(f"Cross-screen registry mismatch: {name}")
        summary = json.loads(
            screen["summary"].read_text(encoding="utf-8")
        )
        results = summary["results"]
        for competitor in screen["competitors"]:
            if results["functional"][competitor] != EXPECTED_PAIRS:
                raise RuntimeError(
                    f"Prior executable gate is incomplete: {competitor}"
                )
            competitors[competitor] = results["cl100k_base"][competitor]
        evidence[name] = {
            "summary": str(screen["summary"].relative_to(REPOSITORY_ROOT)),
            "corpus": str(screen["corpus"].relative_to(REPOSITORY_ROOT)),
            "registry_match": True,
            "exact_stdout": {
                competitor: results["functional"][competitor]
                for competitor in screen["competitors"]
            },
        }
    return {
        "protocol": (
            "Current Kern and frontier runtimes joined to prior pinned "
            "executable screens only after exact task, Python source, and "
            "stdout-oracle registry equality"
        ),
        "kern_cl100k_base_current": current["cl100k_base"]["kern"],
        "competitor_cl100k_base": dict(
            sorted(competitors.items(), key=lambda item: item[1])
        ),
        "prior_screen_evidence": evidence,
    }


def write_details(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_token_svg(path: Path, row: dict[str, Any]) -> None:
    colors = {
        "python": "#94a3b8",
        "python_minifier": "#06b6d4",
        "gnu_apl": "#a855f7",
        "cjam": "#f97316",
        "kona": "#eab308",
        "kern": "#22c55e",
    }
    labels = {
        "python": "Python",
        "python_minifier": "python-minifier",
        "gnu_apl": "GNU APL",
        "cjam": "CJam",
        "kona": "Kona K3",
        "kern": "Kern compact",
    }
    order = (
        "python",
        "python_minifier",
        "gnu_apl",
        "cjam",
        "kona",
        "kern",
    )
    write_grouped_bar_svg(
        path,
        title="Kern crosses the executable token frontier",
        subtitle=(
            "14 matched programs · exact stdout · complete sources · "
            "lower is better"
        ),
        groups=["cl100k_base", "o200k_base"],
        series=[
            (
                labels[name],
                colors[name],
                [
                    row["cl100k_base"][name],
                    row["o200k_base"][name],
                ],
            )
            for name in order
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
        title="Native-system frontier",
        subtitle=(
            "Competitors use cl100k_base; Kern is shown with shared and "
            "frozen native tokenizers"
        ),
        groups=["GNU APL", "CJam", "Kona K3"],
        series=[
            (
                "Competitor + cl100k",
                "#f97316",
                [shared[name] for name in COMPETITORS],
            ),
            (
                "Kern + cl100k",
                "#06b6d4",
                [shared["kern"]] * len(COMPETITORS),
            ),
            (
                "Kern + Kern-16K",
                "#22c55e",
                [native] * len(COMPETITORS),
            ),
        ],
        y_label="Aggregate tokens",
        max_value=180.0,
        value_suffix="",
    )


def write_byte_svg(path: Path, row: dict[str, Any]) -> None:
    values = row["utf8_bytes"]
    write_grouped_bar_svg(
        path,
        title="Complete-source UTF-8 accounting",
        subtitle=(
            "Byte golf is a separate metric; CJam remains ahead in this lane"
        ),
        groups=["14 programs"],
        series=[
            ("Python", "#94a3b8", [values["python"]]),
            ("GNU APL", "#a855f7", [values["gnu_apl"]]),
            ("CJam", "#f97316", [values["cjam"]]),
            ("Kona K3", "#eab308", [values["kona"]]),
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
        title="Frontier functional preservation",
        subtitle="Every source executes against the same normalized stdout",
        groups=["14 matched programs"],
        series=[
            (
                "Kern",
                "#22c55e",
                [functional["kern"] / total * 100],
            ),
            (
                "GNU APL",
                "#a855f7",
                [functional["gnu_apl"] / total * 100],
            ),
            (
                "CJam",
                "#f97316",
                [functional["cjam"] / total * 100],
            ),
            (
                "Kona K3",
                "#eab308",
                [functional["kona"] / total * 100],
            ),
        ],
        y_label="Normalized stdout oracle pass rate (%)",
        max_value=100.0,
    )


def write_market_svg(path: Path, row: dict[str, Any]) -> None:
    labels = {
        "cjam": "CJam",
        "kona": "Kona K3",
        "jelly": "Jelly",
        "pyth": "Pyth",
        "gnu_apl": "GNU APL",
        "j": "J",
        "golfscript": "GolfScript",
        "k": "K",
        "uiua": "Uiua",
        "bqn": "BQN",
    }
    ordered = [("kern", row["kern_cl100k_base_current"])] + list(
        row["competitor_cl100k_base"].items()
    )
    direct = {"cjam", "kona", "gnu_apl"}
    write_grouped_bar_svg(
        path,
        title="Executable compact-language market screen",
        subtitle=(
            "Same 14 tasks and oracles · current Kern; pinned exact-output "
            "competitor screens · lower is better"
        ),
        groups=["Kern"] + [labels[name] for name, _ in ordered[1:]],
        series=[
            (
                "Kern current",
                "#22c55e",
                [value if name == "kern" else None for name, value in ordered],
            ),
            (
                "Current direct run",
                "#f97316",
                [value if name in direct else None for name, value in ordered],
            ),
            (
                "Prior pinned run",
                "#38bdf8",
                [
                    value
                    if name != "kern" and name not in direct
                    else None
                    for name, value in ordered
                ],
            ),
        ],
        y_label="Aggregate LLM tokens",
        max_value=250.0,
        value_suffix="",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnu-apl-binary", type=Path, required=True)
    parser.add_argument(
        "--gnu-apl-source-archive",
        type=Path,
        required=True,
    )
    parser.add_argument("--cjam-jar", type=Path, required=True)
    parser.add_argument("--java", default="java")
    parser.add_argument("--kona-root", type=Path, required=True)
    parser.add_argument("--kona-binary", type=Path)
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
        default=Path("benchmark_results/frontier-languages"),
    )
    args = parser.parse_args()
    args.kona_binary = args.kona_binary or args.kona_root / "k"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        args.gnu_apl_binary,
        args.gnu_apl_source_archive,
        args.cjam_jar,
        args.kona_root / ".git",
        args.kona_binary,
        args.kona_root / "k_test",
        args.tokenizer,
        args.tokenizer_manifest,
    ):
        if not path.exists():
            raise RuntimeError(f"Required runtime artifact is missing: {path}")

    gates = runtime_gates(
        gnu_apl_binary=args.gnu_apl_binary,
        gnu_apl_source_archive=args.gnu_apl_source_archive,
        cjam_jar=args.cjam_jar,
        java=args.java,
        kona_root=args.kona_root,
        kona_binary=args.kona_binary,
    )
    manifest = json.loads(
        args.tokenizer_manifest.read_text(encoding="utf-8")
    )
    tokenizer_hash = sha256_file(args.tokenizer)
    if tokenizer_hash != manifest["tokenizer"]["sha256"]:
        raise RuntimeError("Kern native tokenizer hash does not match manifest.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    pairs = frontier_pairs()
    rows = score_pairs(
        pairs=pairs,
        gnu_apl_binary=args.gnu_apl_binary,
        cjam_jar=args.cjam_jar,
        java=args.java,
        kona_binary=args.kona_binary,
        tokenizer=tokenizer,
    )
    result = aggregate(rows)
    if any(
        result["functional"][name] != EXPECTED_PAIRS
        for name in REPRESENTATIONS
    ):
        raise RuntimeError("At least one frontier representation failed stdout.")
    if result["structural"]["kern_contract_ast"] != EXPECTED_PAIRS:
        raise RuntimeError("Kern compact contract failed in frontier corpus.")
    if result["structural"]["kern_native_exact_roundtrip"] != EXPECTED_PAIRS:
        raise RuntimeError("Kern native tokenizer failed exact round-trip.")
    market = cross_screen_market(pairs, result)

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Fourteen fixed matched executable programs; adversarially "
                "strengthened complete sources; shared production tokenizers; "
                "exact normalized stdout; separate native and byte lanes"
            ),
            "python": platform.python_version(),
            "tiktoken": package_version("tiktoken"),
            "python_minifier": package_version("python-minifier"),
            "tokenizers": package_version("tokenizers"),
            "kern_tokenizer_sha256": tokenizer_hash,
        },
        "discovery_evidence": {
            "source_snapshot_date": "2026-08-01",
            "official_sources": {
                "gnu_apl": "https://www.gnu.org/software/apl/",
                "cjam": (
                    "https://sourceforge.net/p/cjam/code/ci/0.6.5/tree/"
                ),
                "kona": "https://github.com/kevinlawler/kona",
            },
            "interpretation": (
                "This bounded density screen is not evidence that every "
                "program or every competitor solution is globally minimal"
            ),
        },
        "runtime_gates": gates,
        "corpus": {
            "authorship": (
                "Benchmark-authored and adversarially tightened programs "
                "using documented primitives"
            ),
            "normalization": (
                "Collapse display-only whitespace; preserve value tokens "
                "and order"
            ),
            "kona_gcd_sensitivity": (
                "The scored fixed-input Euclidean spelling is 115/113/185; "
                "a general recursive GCD spelling yields 123/121/195"
            ),
            "sources_and_hashes_published": True,
        },
        "results": result,
        "cross_screen_market": market,
    }
    corpus = [
        {
            **asdict(pair),
            "kern": transpile(pair.python.strip(), compact=True).strip() + "\n",
        }
        for pair in pairs
    ]
    (args.output_dir / "frontier-language-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "frontier-language-corpus.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_details(
        rows,
        args.output_dir / "frontier-language-details.csv",
    )
    write_token_svg(
        args.output_dir / "frontier-language-token-density.svg",
        result,
    )
    write_system_svg(
        args.output_dir / "frontier-language-native-system.svg",
        result,
    )
    write_byte_svg(
        args.output_dir / "frontier-language-utf8-bytes.svg",
        result,
    )
    write_functional_svg(
        args.output_dir / "frontier-language-functional.svg",
        result,
    )
    write_market_svg(
        args.output_dir / "frontier-language-market.svg",
        market,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
