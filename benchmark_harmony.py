"""Audit raw-source token density under o200k_base and o200k_harmony.

Harmony extends the o200k encoding with message/control special tokens. This
benchmark deliberately uses ``encode_ordinary`` so language source is measured
as source, never as a synthetic chat envelope. It verifies the encoding
contract and scores the same 1,682 modern programs used by benchmark_modern.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tiktoken
import tiktoken_ext.openai_public as openai_public
from evalplus.data import get_human_eval_plus_hash, get_mbpp_plus_hash

from benchmark_compact_languages import sha256_file, sha256_text, write_grouped_bar_svg
from benchmark_modern import (
    BIGCODEBENCH_REVISION,
    REPRESENTATIONS,
    Task,
    load_bigcodebench_tasks,
    load_evalplus_tasks,
    transform,
)

ENCODINGS = ("cl100k_base", "o200k_base", "o200k_harmony")
EXPECTED_PROGRAMS = 1682


def mergeable_ranks_sha256(encoding: tiktoken.Encoding) -> str:
    """Hash byte tokens and ranks in numeric-rank order."""
    digest = hashlib.sha256()
    for token, rank in sorted(
        encoding._mergeable_ranks.items(), key=lambda item: item[1]
    ):
        digest.update(rank.to_bytes(4, "big"))
        digest.update(len(token).to_bytes(4, "big"))
        digest.update(token)
    return digest.hexdigest()


def encoding_contract() -> dict[str, Any]:
    """Return reproducible evidence for the local Harmony construction."""
    base = tiktoken.get_encoding("o200k_base")
    harmony = tiktoken.get_encoding("o200k_harmony")
    source_path = Path(inspect.getsourcefile(openai_public) or "")
    markers = (
        "<|startoftext|>",
        "<|return|>",
        "<|constrain|>",
        "<|channel|>",
        "<|start|>",
        "<|end|>",
        "<|message|>",
        "<|call|>",
    )
    probes = []
    for marker in markers:
        probes.append(
            {
                "marker": marker,
                "harmony_special_token_id": harmony._special_tokens[marker],
                "harmony_allowed_special_tokens": len(
                    harmony.encode(marker, allowed_special={marker})
                ),
                "o200k_base_ordinary_tokens": len(base.encode_ordinary(marker)),
                "o200k_harmony_ordinary_tokens": len(
                    harmony.encode_ordinary(marker)
                ),
            }
        )
    return {
        "pattern_equal": base._pat_str == harmony._pat_str,
        "mergeable_ranks_equal": (
            base._mergeable_ranks == harmony._mergeable_ranks
        ),
        "mergeable_ranks_sha256": {
            "o200k_base": mergeable_ranks_sha256(base),
            "o200k_harmony": mergeable_ranks_sha256(harmony),
        },
        "vocabulary_size": {
            "o200k_base": base.n_vocab,
            "o200k_harmony": harmony.n_vocab,
        },
        "special_token_count": {
            "o200k_base": len(base._special_tokens),
            "o200k_harmony": len(harmony._special_tokens),
        },
        "tiktoken_constructor_source": "tiktoken_ext/openai_public.py",
        "tiktoken_constructor_source_sha256": sha256_file(source_path),
        "special_token_probes": probes,
    }


def score_case(
    task: Task,
    representation: str,
    encodings: dict[str, tiktoken.Encoding],
) -> dict[str, Any]:
    """Score one transformed source under all ordinary-text encodings."""
    encoded, _ = transform(task.source, representation)
    ids = {
        name: encoding.encode_ordinary(encoded)
        for name, encoding in encodings.items()
    }
    return {
        "dataset": task.dataset,
        "task_id": task.task_id,
        "representation": representation,
        "source_sha256": sha256_text(encoded),
        "utf8_bytes": len(encoded.encode("utf-8")),
        "cl100k_base": len(ids["cl100k_base"]),
        "o200k_base": len(ids["o200k_base"]),
        "o200k_harmony": len(ids["o200k_harmony"]),
        "o200k_token_ids_equal": (
            ids["o200k_base"] == ids["o200k_harmony"]
        ),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate full-denominator token totals and equality gates."""
    datasets = sorted({row["dataset"] for row in rows})
    totals: dict[str, Any] = {}
    for dataset in datasets + ["Combined"]:
        subset = (
            rows if dataset == "Combined"
            else [row for row in rows if row["dataset"] == dataset]
        )
        totals[dataset] = {}
        for representation in REPRESENTATIONS:
            selected = [
                row for row in subset
                if row["representation"] == representation
            ]
            totals[dataset][representation] = {
                "programs": len(selected),
                **{
                    name: sum(row[name] for row in selected)
                    for name in ENCODINGS
                },
                "o200k_exact_id_matches": sum(
                    row["o200k_token_ids_equal"] for row in selected
                ),
            }
    return {
        "programs": len(rows) // len(REPRESENTATIONS),
        "representation_rows": len(rows),
        "o200k_exact_id_matches": sum(
            row["o200k_token_ids_equal"] for row in rows
        ),
        "o200k_token_delta": sum(
            row["o200k_harmony"] - row["o200k_base"] for row in rows
        ),
        "max_absolute_case_delta": max(
            abs(row["o200k_harmony"] - row["o200k_base"])
            for row in rows
        ),
        "totals": totals,
    }


def write_details(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_graph(result: dict[str, Any], path: Path) -> None:
    combined = result["totals"]["Combined"]
    chart_encodings = ("o200k_base", "o200k_harmony")
    labels = {
        "python": "Python",
        "kern": "Kern reversible",
        "kern_compact": "Kern compact",
        "python_minifier": "python-minifier",
    }
    order = ("python", "kern", "kern_compact", "python_minifier")
    maximum = max(
        combined[representation][name]
        for representation in order
        for name in chart_encodings
    )
    write_grouped_bar_svg(
        path,
        title="o200k_harmony raw-source equivalence",
        subtitle=(
            "1,682 code-only programs · encode_ordinary · lower is better"
        ),
        groups=[labels[name] for name in order],
        series=[
            (
                "o200k_base",
                "#38bdf8",
                [combined[item]["o200k_base"] for item in order],
            ),
            (
                "o200k_harmony",
                "#f97316",
                [combined[item]["o200k_harmony"] for item in order],
            ),
        ],
        y_label="Aggregate ordinary-source tokens",
        max_value=float((maximum // 10000 + 2) * 10000),
        value_suffix="",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/harmony"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_evalplus_tasks() + load_bigcodebench_tasks()
    if len(tasks) != EXPECTED_PROGRAMS:
        raise RuntimeError(
            f"Expected {EXPECTED_PROGRAMS} programs, received {len(tasks)}"
        )
    encodings = {
        name: tiktoken.get_encoding(name) for name in ENCODINGS
    }
    rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        for representation in REPRESENTATIONS:
            rows.append(score_case(task, representation, encodings))
        if index % 100 == 0 or index == len(tasks):
            print(f"Scored {index}/{len(tasks)} programs...")

    contract = encoding_contract()
    result = aggregate(rows)
    if not contract["pattern_equal"] or not contract["mergeable_ranks_equal"]:
        raise RuntimeError("Harmony no longer shares the o200k ordinary encoding")
    if result["o200k_exact_id_matches"] != len(rows):
        raise RuntimeError("At least one ordinary-source token stream differs")
    if result["o200k_token_delta"] or result["max_absolute_case_delta"]:
        raise RuntimeError("At least one ordinary-source token count differs")

    source_path = Path(inspect.getsourcefile(openai_public) or "")
    summary = {
        "schema_version": 1,
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": (
                "Code-only source scored with encode_ordinary; no chat or "
                "Harmony message envelope is included"
            ),
            "tiktoken": importlib.metadata.version("tiktoken"),
            "bigcodebench_revision": BIGCODEBENCH_REVISION,
            "bigcodebench_split": "v0.1.4",
            "humaneval_plus_hash": get_human_eval_plus_hash(),
            "mbpp_plus_hash": get_mbpp_plus_hash(),
            "constructor_function": "tiktoken_ext.openai_public.o200k_harmony",
            "constructor_source_file": source_path.name,
        },
        "encoding_contract": contract,
        "results": result,
        "interpretation": (
            "Under tiktoken 0.13.0, o200k_harmony adds special tokens but "
            "uses the same pattern and mergeable ranks as o200k_base; raw "
            "ordinary code therefore has identical token IDs and counts"
        ),
    }
    (args.output_dir / "harmony-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_details(rows, args.output_dir / "harmony-details.csv")
    write_graph(result, args.output_dir / "harmony-token-density.svg")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
