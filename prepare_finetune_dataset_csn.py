"""
prepare_finetune_dataset_csn.py

Build a larger Python -> Kern fine-tuning corpus from CodeSearchNet (Python).
Designed for low-disk workflows:
  - uses HF streaming by default (no full dataset download)
  - keeps only accepted examples in memory

Outputs:
  <out-dir>/
    train/
      py/*.py
      kern/*.kern
      pairs.jsonl
    valid/
      py/*.py
      kern/*.kern
      pairs.jsonl
    train_qwen_chat.jsonl
    valid_qwen_chat.jsonl
    summary.json
    rejected_sample.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from kern_compiler import compile_kern
from kern_transpiler import transpile


UNSUPPORTED_STMT = "# UNSUPPORTED:"
UNSUPPORTED_EXPR_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_]*>")


@dataclass
class SourceRow:
    source: str
    repo: str
    func_name: str
    func_path: str
    raw_id: str


@dataclass
class Sample:
    dataset: str
    task_id: str
    python: str
    kern: str
    repo: str
    func_name: str
    func_path: str


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def iter_codesearchnet_python(split: str, streaming: bool) -> SourceRow:
    ds = load_dataset("code_search_net", "python", split=split, streaming=streaming)
    for i, item in enumerate(ds):
        source = (item.get("whole_func_string") or item.get("func_code_string") or "").strip()
        if not source:
            continue
        repo = str(item.get("repository_name") or "")
        func_name = str(item.get("func_name") or "")
        func_path = str(item.get("func_path_in_repository") or "")
        raw_id = f"csn/python/{split}/{i}"
        yield SourceRow(
            source=source,
            repo=repo,
            func_name=func_name,
            func_path=func_path,
            raw_id=raw_id,
        )


def has_def(module: ast.AST) -> bool:
    if not isinstance(module, ast.Module):
        return False
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
    return False


def parse_python_quiet(src: str) -> ast.AST:
    # Large real-world corpora often contain string literals with backslashes
    # that trigger SyntaxWarning on recent Python versions; keep logs readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(src)


def quality_check(
    src: str,
    min_chars: int,
    max_chars: int,
    min_lines: int,
    max_lines: int,
) -> tuple[bool, str]:
    n_chars = len(src)
    if n_chars < min_chars:
        return False, "too_short"
    if n_chars > max_chars:
        return False, "too_long"

    n_lines = src.count("\n") + 1
    if n_lines < min_lines:
        return False, "too_few_lines"
    if n_lines > max_lines:
        return False, "too_many_lines"

    if "\x00" in src:
        return False, "contains_nul"

    if src.count("\n\n\n\n") > 0:
        return False, "suspicious_spacing"

    try:
        tree = parse_python_quiet(src)
    except Exception:
        return False, "python_parse_fail"

    if not has_def(tree):
        return False, "no_function_def"

    return True, "ok"


def stable_task_id(row: SourceRow, accepted_idx: int) -> str:
    base = f"{row.repo}|{row.func_path}|{row.func_name}|{row.raw_id}|{accepted_idx}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    token = sanitize_id(f"csn_python__{row.repo}__{row.func_name}__{digest}")
    if not token:
        token = f"csn_python__{digest}"
    return token


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


def write_split(base: Path, split: str, rows: list[Sample]) -> Path:
    split_dir = base / split
    py_dir = split_dir / "py"
    kern_dir = split_dir / "kern"
    py_dir.mkdir(parents=True, exist_ok=True)
    kern_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = split_dir / "pairs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in rows:
            sid = sanitize_id(row.task_id)
            py_path = py_dir / f"{sid}.py"
            kern_path = kern_dir / f"{sid}.kern"

            py_path.write_text(row.python, encoding="utf-8")
            kern_path.write_text(row.kern, encoding="utf-8")

            record = {
                "id": sid,
                "dataset": row.dataset,
                "task_id": row.task_id,
                "repo": row.repo,
                "func_name": row.func_name,
                "func_path": row.func_path,
                "python_path": str(py_path.relative_to(base)),
                "kern_path": str(kern_path.relative_to(base)),
                "python": row.python,
                "kern": row.kern,
            }
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
    return jsonl_path


def write_qwen_chat_from_pairs(pairs_path: Path, out_path: Path) -> int:
    count = 0
    with pairs_path.open("r", encoding="utf-8") as rf, out_path.open("w", encoding="utf-8") as wf:
        for line in rf:
            if not line.strip():
                continue
            row = json.loads(line)
            rec = {
                "id": row["id"],
                "dataset": row.get("dataset", "codesearchnet_python"),
                "task_id": row.get("task_id", ""),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres experto en Kern. Convierte Python a Kern compacto y valido. "
                            "Responde solo con codigo Kern."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Convierte este Python a Kern:\n\n```python\n{row['python']}\n```",
                    },
                    {"role": "assistant", "content": row["kern"]},
                ],
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a high-quality Python->Kern dataset from CodeSearchNet (Python)."
    )
    parser.add_argument("--out-dir", default="data/finetune_csn20k", help="Output directory.")
    parser.add_argument("--split", default="train", help="CodeSearchNet split to scan.")
    parser.add_argument("--target-kept", type=int, default=20000, help="Accepted samples target.")
    parser.add_argument("--scan-limit", type=int, default=0, help="Max scanned rows (0 = unlimited).")
    parser.add_argument("--valid-ratio", type=float, default=0.05, help="Validation split ratio in [0,1].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--min-chars", type=int, default=80, help="Minimum source length.")
    parser.add_argument("--max-chars", type=int, default=5000, help="Maximum source length.")
    parser.add_argument("--min-lines", type=int, default=4, help="Minimum source lines.")
    parser.add_argument("--max-lines", type=int, default=220, help="Maximum source lines.")
    parser.add_argument(
        "--rejected-sample-limit",
        type=int,
        default=2000,
        help="Store up to this many rejected examples for inspection.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N scanned rows.",
    )
    parser.add_argument("--no-streaming", action="store_true", help="Disable HF streaming mode.")
    parser.add_argument("--no-validate", action="store_true", help="Skip compile/parse validation.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before writing.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    kept: list[Sample] = []
    rejected_sample: list[dict[str, str]] = []
    reject_counts: Counter[str] = Counter()
    seen_hashes: set[str] = set()

    for row in iter_codesearchnet_python(split=args.split, streaming=not args.no_streaming):
        if args.scan_limit > 0 and seen >= args.scan_limit:
            break
        seen += 1

        norm = re.sub(r"\s+", " ", row.source).strip()
        source_hash = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if source_hash in seen_hashes:
            reject_counts["duplicate"] += 1
            continue
        seen_hashes.add(source_hash)

        ok, reason = quality_check(
            row.source,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_lines=args.min_lines,
            max_lines=args.max_lines,
        )
        if not ok:
            reject_counts[reason] += 1
            if len(rejected_sample) < args.rejected_sample_limit:
                rejected_sample.append(
                    {
                        "raw_id": row.raw_id,
                        "repo": row.repo,
                        "func_name": row.func_name,
                        "func_path": row.func_path,
                        "stage": "quality",
                        "reason": reason,
                    }
                )
            continue

        try:
            kern_src = transpile(row.source)
        except Exception as exc:  # noqa: BLE001
            reject_counts["transpile_fail"] += 1
            if len(rejected_sample) < args.rejected_sample_limit:
                rejected_sample.append(
                    {
                        "raw_id": row.raw_id,
                        "repo": row.repo,
                        "func_name": row.func_name,
                        "func_path": row.func_path,
                        "stage": "transpile",
                        "reason": str(exc),
                    }
                )
            continue

        if UNSUPPORTED_STMT in kern_src or UNSUPPORTED_EXPR_RE.search(kern_src):
            reject_counts["unsupported_marker"] += 1
            continue

        if not args.no_validate:
            try:
                py_back = compile_kern(kern_src)
                parse_python_quiet(py_back)
            except Exception as exc:  # noqa: BLE001
                reject_counts["compile_parse_fail"] += 1
                if len(rejected_sample) < args.rejected_sample_limit:
                    rejected_sample.append(
                        {
                            "raw_id": row.raw_id,
                            "repo": row.repo,
                            "func_name": row.func_name,
                            "func_path": row.func_path,
                            "stage": "compile_parse",
                            "reason": str(exc),
                        }
                    )
                continue

        task_id = stable_task_id(row, accepted_idx=len(kept))
        kept.append(
            Sample(
                dataset="codesearchnet_python",
                task_id=task_id,
                python=row.source,
                kern=kern_src,
                repo=row.repo,
                func_name=row.func_name,
                func_path=row.func_path,
            )
        )

        if args.progress_every > 0 and seen % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "progress_scanned": seen,
                        "progress_kept": len(kept),
                        "progress_rejected": sum(reject_counts.values()),
                    }
                )
            )

        if len(kept) >= args.target_kept:
            break

    train_rows, valid_rows = split_samples(kept, valid_ratio=args.valid_ratio, seed=args.seed)
    train_pairs = write_split(out_dir, "train", train_rows)
    valid_pairs = write_split(out_dir, "valid", valid_rows)
    train_chat_count = write_qwen_chat_from_pairs(train_pairs, out_dir / "train_qwen_chat.jsonl")
    valid_chat_count = write_qwen_chat_from_pairs(valid_pairs, out_dir / "valid_qwen_chat.jsonl")

    summary = {
        "source_dataset": "code_search_net/python",
        "split": args.split,
        "streaming": not args.no_streaming,
        "validate_roundtrip": not args.no_validate,
        "target_kept": args.target_kept,
        "scan_limit": args.scan_limit,
        "scanned": seen,
        "kept": len(kept),
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "train_qwen_chat_count": train_chat_count,
        "valid_qwen_chat_count": valid_chat_count,
        "keep_rate": (len(kept) / seen) if seen else 0.0,
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "quality_filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_lines": args.min_lines,
            "max_lines": args.max_lines,
        },
        "rejected_counts": dict(sorted(reject_counts.items(), key=lambda kv: kv[0])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "rejected_sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected_sample),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Written dataset to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
