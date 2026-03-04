"""
prepare_finetune_dataset.py

Builds a fine-tuning corpus from Python benchmarks into:
  - visible files: .py and .kern
  - trainer-friendly file: pairs.jsonl

Supported sources:
  - HumanEval
  - MBPP train
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from human_eval.data import read_problems

from kern_compiler import compile_kern
from kern_transpiler import transpile


UNSUPPORTED_STMT = "# UNSUPPORTED:"
UNSUPPORTED_EXPR_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")


@dataclass
class Sample:
    dataset: str
    task_id: str
    python: str
    kern: str


def load_humaneval() -> list[tuple[str, str, str]]:
    problems = read_problems()
    rows: list[tuple[str, str, str]] = []
    for task_id, p in problems.items():
        source = p["prompt"] + p["canonical_solution"]
        rows.append(("humaneval", task_id, source))
    return rows


def load_mbpp_train() -> list[tuple[str, str, str]]:
    ds = load_dataset("mbpp", split="train")
    rows: list[tuple[str, str, str]] = []
    for item in ds:
        rows.append(("mbpp_train", f"MBPP/{item['task_id']}", item["code"]))
    return rows


def sanitize_id(dataset: str, task_id: str) -> str:
    base = f"{dataset}__{task_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def make_samples(
    datasets: list[str],
    max_cases: int,
    validate_roundtrip: bool,
) -> tuple[list[Sample], list[dict[str, str]]]:
    rows: list[tuple[str, str, str]] = []
    if "humaneval" in datasets:
        rows.extend(load_humaneval())
    if "mbpp_train" in datasets:
        rows.extend(load_mbpp_train())
    if max_cases > 0:
        rows = rows[:max_cases]

    out: list[Sample] = []
    rejected: list[dict[str, str]] = []

    for dataset, task_id, py_src in rows:
        try:
            kern_src = transpile(py_src)
        except Exception as exc:  # noqa: BLE001
            rejected.append(
                {
                    "dataset": dataset,
                    "task_id": task_id,
                    "stage": "transpile",
                    "error": str(exc),
                }
            )
            continue

        if UNSUPPORTED_STMT in kern_src or UNSUPPORTED_EXPR_RE.search(kern_src):
            rejected.append(
                {
                    "dataset": dataset,
                    "task_id": task_id,
                    "stage": "transpile",
                    "error": "unsupported marker emitted",
                }
            )
            continue

        if validate_roundtrip:
            try:
                py_back = compile_kern(kern_src)
                ast.parse(py_back)
            except Exception as exc:  # noqa: BLE001
                rejected.append(
                    {
                        "dataset": dataset,
                        "task_id": task_id,
                        "stage": "compile/parse",
                        "error": str(exc),
                    }
                )
                continue

        out.append(Sample(dataset=dataset, task_id=task_id, python=py_src, kern=kern_src))

    return out, rejected


def split_samples(samples: list[Sample], valid_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    items = list(samples)
    rng = random.Random(seed)
    rng.shuffle(items)

    if valid_ratio <= 0.0:
        return items, []
    if valid_ratio >= 1.0:
        return [], items

    n_valid = int(len(items) * valid_ratio)
    if n_valid == 0 and items:
        n_valid = 1
    n_valid = min(n_valid, len(items))

    valid = items[:n_valid]
    train = items[n_valid:]
    return train, valid


def write_split(base: Path, split: str, rows: list[Sample]) -> None:
    split_dir = base / split
    py_dir = split_dir / "py"
    kern_dir = split_dir / "kern"
    py_dir.mkdir(parents=True, exist_ok=True)
    kern_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = split_dir / "pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            sid = sanitize_id(row.dataset, row.task_id)
            py_path = py_dir / f"{sid}.py"
            kern_path = kern_dir / f"{sid}.kern"

            py_path.write_text(row.python, encoding="utf-8")
            kern_path.write_text(row.kern, encoding="utf-8")

            record = {
                "id": sid,
                "dataset": row.dataset,
                "task_id": row.task_id,
                "python_path": str(py_path.relative_to(base)),
                "kern_path": str(kern_path.relative_to(base)),
                "python": row.python,
                "kern": row.kern,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build .kern + JSONL fine-tuning dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval", "mbpp_train"],
        choices=["humaneval", "mbpp_train"],
        help="Input datasets to include.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/finetune",
        help="Output directory.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional cap on total loaded examples (0 = all).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip compile/parse validation of Kern output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output directory before writing.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, rejected = make_samples(
        datasets=args.datasets,
        max_cases=args.max_cases,
        validate_roundtrip=not args.no_validate,
    )

    train_rows, valid_rows = split_samples(samples, valid_ratio=args.valid_ratio, seed=args.seed)
    write_split(out_dir, "train", train_rows)
    write_split(out_dir, "valid", valid_rows)

    summary = {
        "datasets": args.datasets,
        "validate_roundtrip": not args.no_validate,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "max_cases": args.max_cases,
        "total_kept": len(samples),
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "rejected_count": len(rejected),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Written dataset to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
