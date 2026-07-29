"""Held-out benchmark for Kern's native 16K BPE and Toke's native BPE.

The modern lane scores Kern compact on HumanEval+, MBPP+, and BigCodeBench.
The paired lane scores Kern and Toke on the 60 equivalent public JSON-CLI
programs from the pinned toke-eval repository.  Shared-tokenizer and native
results remain explicitly labelled because the source languages differ.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import statistics
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as svg_escape

import python_minifier
import tiktoken
from tokenizers import Tokenizer

from benchmark_modern import (
    TOKENIZERS,
    load_bigcodebench_tasks,
    load_evalplus_tasks,
    write_grouped_bar_svg,
)
from benchmark_toke import (
    TOKE_COMMIT,
    TOKE_EVAL_COMMIT,
    TOKE_TOKENIZER_VERSION,
    TOKE_TOKENIZER_WHEEL_SHA256,
    count_toke_tokens,
    load_pairs,
)
from kern_transpiler import transpile
from train_kern_tokenizer import sha256_file, sha256_text

REPRESENTATIONS = (
    "python_cl100k",
    "kern_cl100k",
    "kern_native_16k",
    "python_minifier_cl100k",
)


@dataclass(frozen=True)
class NativeResult:
    dataset: str
    task_id: str
    python_sha256: str
    python_cl100k: int
    kern_cl100k: int
    kern_native_16k: int
    python_minifier_cl100k: int
    kern_native_exact_roundtrip: bool
    toke_sha256: str = ""
    toke_cl100k: int | None = None
    toke_native_16k: int | None = None


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def pct_below(value: int, baseline: int) -> float:
    return (baseline - value) / baseline * 100 if baseline else 0.0


def score_kern_source(
    *,
    dataset: str,
    task_id: str,
    python_source: str,
    kern_source: str,
    minified_source: str,
    native: Tokenizer,
    cl100k: Any,
    toke_source: str = "",
) -> NativeResult:
    ids = native.encode(kern_source).ids
    return NativeResult(
        dataset=dataset,
        task_id=task_id,
        python_sha256=sha256_text(python_source),
        python_cl100k=len(cl100k.encode(python_source)),
        kern_cl100k=len(cl100k.encode(kern_source)),
        kern_native_16k=len(ids),
        python_minifier_cl100k=len(cl100k.encode(minified_source)),
        kern_native_exact_roundtrip=native.decode(ids) == kern_source,
        toke_sha256=sha256_text(toke_source) if toke_source else "",
        toke_cl100k=(
            len(cl100k.encode(toke_source)) if toke_source else None
        ),
        toke_native_16k=(
            count_toke_tokens(toke_source) if toke_source else None
        ),
    )


def aggregate_modern(
    results: list[NativeResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted(
        {
            result.dataset
            for result in results
            if result.dataset != "Toke public pairs"
        }
    )
    for dataset in (*datasets, "Combined"):
        subset = [
            result
            for result in results
            if result.dataset != "Toke public pairs"
            and (dataset == "Combined" or result.dataset == dataset)
        ]
        python_tokens = sum(result.python_cl100k for result in subset)
        kern_shared = sum(result.kern_cl100k for result in subset)
        kern_native = sum(result.kern_native_16k for result in subset)
        minifier = sum(
            result.python_minifier_cl100k for result in subset
        )
        rows.append(
            {
                "dataset": dataset,
                "programs": len(subset),
                "python_cl100k": python_tokens,
                "kern_cl100k": kern_shared,
                "kern_native_16k": kern_native,
                "python_minifier_cl100k": minifier,
                "kern_cl100k_saved_vs_python_pct": pct_below(
                    kern_shared, python_tokens
                ),
                "kern_native_saved_vs_python_cl100k_pct": pct_below(
                    kern_native, python_tokens
                ),
                "kern_native_below_kern_cl100k_pct": pct_below(
                    kern_native, kern_shared
                ),
                "kern_native_below_minifier_cl100k_pct": pct_below(
                    kern_native, minifier
                ),
                "native_exact_roundtrips": sum(
                    result.kern_native_exact_roundtrip for result in subset
                ),
            }
        )
    return rows


def aggregate_pairs(results: list[NativeResult]) -> dict[str, Any]:
    pairs = [
        result
        for result in results
        if result.dataset == "Toke public pairs"
    ]
    python_tokens = sum(result.python_cl100k for result in pairs)
    kern_shared = sum(result.kern_cl100k for result in pairs)
    kern_native = sum(result.kern_native_16k for result in pairs)
    minifier = sum(result.python_minifier_cl100k for result in pairs)
    toke_shared = sum(result.toke_cl100k or 0 for result in pairs)
    toke_native = sum(result.toke_native_16k or 0 for result in pairs)
    per_pair_advantages = [
        pct_below(result.kern_native_16k, result.toke_native_16k or 0)
        for result in pairs
    ]
    return {
        "programs": len(pairs),
        "python_cl100k": python_tokens,
        "kern_cl100k": kern_shared,
        "kern_native_16k": kern_native,
        "python_minifier_cl100k": minifier,
        "toke_cl100k": toke_shared,
        "toke_native_16k": toke_native,
        "kern_native_below_toke_native_pct": pct_below(
            kern_native, toke_native
        ),
        "kern_native_below_toke_native_pairs": sum(
            result.kern_native_16k < (result.toke_native_16k or 0)
            for result in pairs
        ),
        "median_per_pair_kern_below_toke_native_pct": statistics.median(
            per_pair_advantages
        ),
        "kern_native_exact_roundtrips": sum(
            result.kern_native_exact_roundtrip for result in pairs
        ),
    }


def write_details(results: list[NativeResult], path: Path) -> None:
    fields = list(asdict(results[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_paired_svg(path: Path, pairs: dict[str, Any]) -> None:
    bars = [
        ("Python + cl100k", pairs["python_cl100k"], "#94a3b8"),
        ("python-minifier + cl100k", pairs["python_minifier_cl100k"], "#06b6d4"),
        ("Toke + native 16K", pairs["toke_native_16k"], "#f59e0b"),
        ("Kern + native 16K", pairs["kern_native_16k"], "#22c55e"),
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
        "<title id=\"title\">Held-out native-tokenizer contest</title>",
        (
            "<desc id=\"desc\">Aggregate tokens on 60 equivalent public "
            "JSON-CLI program pairs; lower is better</desc>"
        ),
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        (
            f'<text x="{left}" y="38" fill="#f8fafc" '
            'font-family="Inter,system-ui,sans-serif" font-size="24" '
            'font-weight="700">Held-out native-tokenizer contest</text>'
        ),
        (
            f'<text x="{left}" y="66" fill="#94a3b8" '
            'font-family="Inter,system-ui,sans-serif" font-size="14">'
            "60 equivalent public JSON-CLI pairs; lower is better</text>"
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


def lookup(
    rows: list[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    return next(row for row in rows if row["dataset"] == dataset)


def write_modern_svg(
    path: Path,
    aggregates: list[dict[str, Any]],
) -> None:
    groups = ["HumanEval+", "MBPP+", "BigCodeBench"]
    write_grouped_bar_svg(
        path,
        title="Held-out Kern token reduction",
        subtitle=(
            "Kern native 16K is trained on disjoint CodeSearchNet; "
            "comparators use cl100k_base"
        ),
        groups=groups,
        series=[
            (
                "Kern + native 16K",
                "#22c55e",
                [
                    lookup(aggregates, group)[
                        "kern_native_saved_vs_python_cl100k_pct"
                    ]
                    for group in groups
                ],
            ),
            (
                "Kern + cl100k",
                "#7c3aed",
                [
                    lookup(aggregates, group)[
                        "kern_cl100k_saved_vs_python_pct"
                    ]
                    for group in groups
                ],
            ),
            (
                "python-minifier + cl100k",
                "#06b6d4",
                [
                    pct_below(
                        lookup(aggregates, group)[
                            "python_minifier_cl100k"
                        ],
                        lookup(aggregates, group)["python_cl100k"],
                    )
                    for group in groups
                ],
            ),
        ],
        y_label="Tokens saved vs Python + cl100k (%)",
        max_value=70.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "benchmark_results/native-tokenizer/kern-16k-tokenizer.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "benchmark_results/native-tokenizer/"
            "kern-16k-training-manifest.json"
        ),
    )
    parser.add_argument(
        "--toke-eval",
        type=Path,
        required=True,
        help="Pinned toke-eval checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/native-tokenizer"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    actual_sha = sha256_file(args.tokenizer)
    expected_sha = manifest["tokenizer"]["sha256"]
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Kern tokenizer hash mismatch: {actual_sha} != {expected_sha}"
        )
    native = Tokenizer.from_file(str(args.tokenizer))
    if native.get_vocab_size(with_added_tokens=True) != 16_384:
        raise RuntimeError("Kern native tokenizer is not exactly 16K.")
    cl100k = tiktoken.get_encoding("cl100k_base")

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    results: list[NativeResult] = []
    tasks = load_evalplus_tasks() + load_bigcodebench_tasks()
    for index, task in enumerate(tasks, start=1):
        kern = transpile(task.source, compact=True).strip()
        minified = python_minifier.minify(
            task.source,
            rename_globals=False,
        )
        results.append(
            score_kern_source(
                dataset=task.dataset,
                task_id=task.task_id,
                python_source=task.source,
                kern_source=kern,
                minified_source=minified,
                native=native,
                cl100k=cl100k,
            )
        )
        if index % 250 == 0:
            print(f"Scored {index}/{len(tasks)} modern programs...")

    pairs = load_pairs(args.toke_eval)
    for pair in pairs:
        results.append(
            score_kern_source(
                dataset="Toke public pairs",
                task_id=pair.task_id,
                python_source=pair.python,
                kern_source=pair.kern_compact,
                minified_source=pair.python_minifier,
                toke_source=pair.toke,
                native=native,
                cl100k=cl100k,
            )
        )

    if not all(result.kern_native_exact_roundtrip for result in results):
        raise RuntimeError("Kern native tokenizer failed exact source round-trip.")

    modern = aggregate_modern(results)
    pair_summary = aggregate_pairs(results)
    if pair_summary["programs"] != 60:
        raise RuntimeError("The native paired lane must contain 60 programs.")

    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Kern 16K BPE trained on CodeSearchNet train; "
                "pre-tokenizer selected on CodeSearchNet validation; "
                "final evaluation on excluded modern suites and 60 public "
                "Toke pairs"
            ),
            "python": platform.python_version(),
            "tokenizers_version": package_version("tokenizers"),
            "tiktoken_version": package_version("tiktoken"),
            "python_minifier_version": package_version("python-minifier"),
            "toke_tokenizer_version": TOKE_TOKENIZER_VERSION,
            "kern_tokenizer_sha256": actual_sha,
            "kern_vocab_size": native.get_vocab_size(
                with_added_tokens=True
            ),
            "toke_vocab_size": 16_384,
            "shared_tokenizers": list(TOKENIZERS),
            "toke_commit": TOKE_COMMIT,
            "toke_eval_commit": TOKE_EVAL_COMMIT,
            "toke_tokenizer_wheel_sha256": (
                TOKE_TOKENIZER_WHEEL_SHA256
            ),
        },
        "modern_held_out": modern,
        "toke_public_pairs_held_out": pair_summary,
    }
    summary_path = args.output_dir / "native-tokenizer-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_details(
        results,
        args.output_dir / "native-tokenizer-details.csv",
    )
    write_modern_svg(
        args.output_dir / "native-tokenizer-modern.svg",
        modern,
    )
    write_paired_svg(
        args.output_dir / "native-tokenizer-toke.svg",
        pair_summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
