"""
benchmark_head_to_head.py

Unified evaluation harness for representation-level comparisons:
  Python vs Kern vs optional external representations (e.g., SimPy, Token Sugar)

Default adapters:
  - python: identity (baseline)
  - kern:   python -> kern_transpiler.transpile -> kern_compiler.compile_kern

Optional external adapters are loaded from JSON and executed through stdin/stdout:
[
  {
    "name": "simpy",
    "encode_cmd": "python /path/to/encode_simpy.py",
    "decode_cmd": "python /path/to/decode_simpy.py",   // optional
    "python_compatible": false                          // optional; default false
  }
]

If decode_cmd is missing and python_compatible=true, encoded text is treated as executable Python.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import tiktoken
from datasets import load_dataset
from human_eval.data import read_problems
from human_eval.execution import check_correctness

from kern_compiler import compile_kern
from kern_transpiler import transpile


# ---------------------------- datasets ---------------------------------

def load_humaneval() -> list[tuple[str, str, dict]]:
    rows: list[tuple[str, str, dict]] = []
    problems = read_problems()
    for task_id, p in problems.items():
        source = p["prompt"] + p["canonical_solution"]
        rows.append(("humaneval", task_id, {"python_source": source, "problem": p}))
    return rows


def load_mbpp_train() -> list[tuple[str, str, dict]]:
    ds = load_dataset("mbpp", split="train")
    rows: list[tuple[str, str, dict]] = []
    for item in ds:
        rows.append(
            (
                "mbpp_train",
                f"MBPP/{item['task_id']}",
                {"python_source": item["code"], "problem": None},
            )
        )
    return rows


# --------------------------- tokenizers --------------------------------

def build_tokenizers(enabled: list[str]) -> dict[str, Callable[[str], int]]:
    built: dict[str, Callable[[str], int]] = {}
    for name in enabled:
        if name == "cl100k_base":
            enc = tiktoken.get_encoding("cl100k_base")
            built[name] = lambda s, e=enc: len(e.encode(s))
        elif name == "o200k_base":
            enc = tiktoken.get_encoding("o200k_base")
            built[name] = lambda s, e=enc: len(e.encode(s))
        elif name == "llama_tinyllama":
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from transformers import AutoTokenizer  # lazy import
            tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            built[name] = lambda s, t=tok: len(t.encode(s, add_special_tokens=False))
        elif name == "codegen_350m_mono":
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            from transformers import AutoTokenizer  # lazy import
            tok = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
            built[name] = lambda s, t=tok: len(t.encode(s, add_special_tokens=False))
        else:
            raise ValueError(f"Unknown tokenizer: {name}")
    return built


# --------------------------- adapters ----------------------------------

class AdapterError(RuntimeError):
    pass


@dataclass
class RepresentationAdapter:
    name: str
    encode: Callable[[str], str]
    decode_to_python: Callable[[str], str]


def _run_pipe_cmd(cmd: str, text: str, timeout: float = 30.0) -> str:
    proc = subprocess.run(
        cmd,
        input=text,
        text=True,
        capture_output=True,
        shell=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AdapterError(
            f"command failed ({proc.returncode}): {cmd}\n"
            f"stderr: {proc.stderr.strip()[:300]}"
        )
    return proc.stdout


def python_adapter() -> RepresentationAdapter:
    return RepresentationAdapter(
        name="python",
        encode=lambda src: src,
        decode_to_python=lambda encoded: encoded,
    )


def kern_adapter() -> RepresentationAdapter:
    return RepresentationAdapter(
        name="kern",
        encode=lambda src: transpile(src),
        decode_to_python=lambda encoded: compile_kern(encoded),
    )


def load_external_adapters(path: str | None) -> list[RepresentationAdapter]:
    if not path:
        return []
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    adapters: list[RepresentationAdapter] = []
    for item in cfg:
        name = item["name"]
        encode_cmd = item["encode_cmd"]
        decode_cmd = item.get("decode_cmd")
        py_compatible = bool(item.get("python_compatible", False))

        def _make_encode(cmd: str) -> Callable[[str], str]:
            return lambda src: _run_pipe_cmd(cmd, src)

        if decode_cmd:
            decode_fn = lambda encoded, cmd=decode_cmd: _run_pipe_cmd(cmd, encoded)
        elif py_compatible:
            decode_fn = lambda encoded: encoded
        else:
            raise ValueError(
                f"external adapter {name!r} requires decode_cmd or python_compatible=true"
            )

        adapters.append(
            RepresentationAdapter(
                name=name,
                encode=_make_encode(encode_cmd),
                decode_to_python=decode_fn,
            )
        )
    return adapters


def _install_simpy_tree_sitter_compat() -> None:
    """
    SimPy depends on legacy tree_sitter API:
      - Language(path, name)
      - Parser.set_language(lang)
    Recent tree_sitter versions changed both APIs.
    This compatibility shim keeps SimPy usable in this harness.
    """
    import ctypes
    import tree_sitter

    if getattr(tree_sitter, "_kern_simpy_compat", False):
        return

    original_language = tree_sitter.Language

    def compat_language(path_or_ptr, name=None):
        if name is None:
            return original_language(path_or_ptr)
        try:
            # Works in older tree_sitter versions
            return original_language(path_or_ptr, name)
        except TypeError:
            # Newer API needs a pointer. Resolve symbol from shared library.
            lib = ctypes.cdll.LoadLibrary(path_or_ptr)
            symbol = getattr(lib, f"tree_sitter_{name}")
            symbol.restype = ctypes.c_void_p
            return original_language(symbol())

    if not hasattr(tree_sitter.Parser, "set_language"):
        def _set_language(self, language):
            self.language = language
        setattr(tree_sitter.Parser, "set_language", _set_language)

    warnings.filterwarnings(
        "ignore",
        message="int argument support is deprecated",
        category=DeprecationWarning,
    )
    tree_sitter.Language = compat_language
    tree_sitter._kern_simpy_compat = True


def simpy_adapter(simpy_root: str) -> RepresentationAdapter:
    _install_simpy_tree_sitter_compat()
    root = str(Path(simpy_root).resolve())
    expected_files = [
        Path(root) / "spy" / "build" / "python-languages.so",
        Path(root) / "spy" / "build" / "spython-languages.so",
    ]
    missing = [str(p) for p in expected_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "SimPy parser libraries missing. Expected:\n  - "
            + "\n  - ".join(missing)
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from spy import Transformer  # type: ignore

    transformer = Transformer()
    return RepresentationAdapter(
        name="simpy",
        encode=lambda src: transformer.parse(src),
        decode_to_python=lambda encoded: transformer.decode(encoded),
    )


def token_sugar_adapter(token_sugar_root: str, pattern_file: str) -> RepresentationAdapter:
    root = str(Path(token_sugar_root).resolve())
    pattern_path = str(Path(pattern_file).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval import Parser as SugarParser  # type: ignore

    parser = SugarParser()
    parser.set_patterns(pattern_path)
    return RepresentationAdapter(
        name="token_sugar",
        encode=lambda src: parser.encode(src),
        decode_to_python=lambda encoded: parser.parse(encoded)[0],
    )


# ---------------------------- evaluation -------------------------------

@dataclass
class CaseResult:
    dataset: str
    task_id: str
    representation: str
    encode_ok: bool
    decode_ok: bool
    parse_ok: bool
    functional_ok: bool
    functional_applicable: bool
    error_stage: str
    error_message: str
    token_stats: dict[str, dict]


def evaluate_case(
    adapter: RepresentationAdapter,
    dataset: str,
    task_id: str,
    payload: dict,
    tokenizers: dict[str, Callable[[str], int]],
    run_functional: bool,
    timeout: float,
) -> CaseResult:
    source = payload["python_source"]
    problem = payload["problem"]

    try:
        encoded = adapter.encode(source)
    except Exception as exc:  # noqa: BLE001 - keep harness resilient
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            representation=adapter.name,
            encode_ok=False,
            decode_ok=False,
            parse_ok=False,
            functional_ok=False,
            functional_applicable=run_functional and dataset == "humaneval",
            error_stage="encode",
            error_message=str(exc),
            token_stats={},
        )

    try:
        py_back = adapter.decode_to_python(encoded)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            representation=adapter.name,
            encode_ok=True,
            decode_ok=False,
            parse_ok=False,
            functional_ok=False,
            functional_applicable=run_functional and dataset == "humaneval",
            error_stage="decode",
            error_message=str(exc),
            token_stats={},
        )

    try:
        ast.parse(py_back)
    except Exception as exc:  # noqa: BLE001
        return CaseResult(
            dataset=dataset,
            task_id=task_id,
            representation=adapter.name,
            encode_ok=True,
            decode_ok=True,
            parse_ok=False,
            functional_ok=False,
            functional_applicable=run_functional and dataset == "humaneval",
            error_stage="parse_back",
            error_message=str(exc),
            token_stats={},
        )

    functional_applicable = run_functional and dataset == "humaneval"
    functional_ok = False
    if functional_applicable:
        eval_problem = {
            "task_id": problem["task_id"],
            "prompt": "",
            "test": problem["test"],
            "entry_point": problem["entry_point"],
        }
        result = check_correctness(eval_problem, py_back, timeout=timeout)
        functional_ok = bool(result["passed"])
        if not functional_ok:
            return CaseResult(
                dataset=dataset,
                task_id=task_id,
                representation=adapter.name,
                encode_ok=True,
                decode_ok=True,
                parse_ok=True,
                functional_ok=False,
                functional_applicable=True,
                error_stage="functional",
                error_message=str(result.get("result", "failed")),
                token_stats={},
            )

    stats: dict[str, dict] = {}
    for tok_name, tok_len in tokenizers.items():
        py_t = tok_len(source)
        rep_t = tok_len(encoded)
        saved = py_t - rep_t
        pct = (saved / py_t) * 100 if py_t else 0.0
        stats[tok_name] = {
            "python_tokens": py_t,
            "repr_tokens": rep_t,
            "saved_tokens": saved,
            "saved_pct": pct,
        }

    return CaseResult(
        dataset=dataset,
        task_id=task_id,
        representation=adapter.name,
        encode_ok=True,
        decode_ok=True,
        parse_ok=True,
        functional_ok=functional_ok if functional_applicable else False,
        functional_applicable=functional_applicable,
        error_stage="",
        error_message="",
        token_stats=stats,
    )


def summarize(
    results: list[CaseResult],
    tokenizers: list[str],
    datasets: list[str],
) -> list[dict]:
    rows: list[dict] = []
    reps = sorted({r.representation for r in results})
    for rep in reps:
        for ds in datasets + ["overall"]:
            if ds == "overall":
                subset = [r for r in results if r.representation == rep]
            else:
                subset = [r for r in results if r.representation == rep and r.dataset == ds]

            total = len(subset)
            parsed = [r for r in subset if r.parse_ok]
            parsed_n = len(parsed)
            functional_subset = [r for r in subset if r.functional_applicable]
            functional_total = len(functional_subset)
            functional_n = sum(1 for r in functional_subset if r.functional_ok)

            for tok in tokenizers:
                py_total = sum(r.token_stats[tok]["python_tokens"] for r in parsed if tok in r.token_stats)
                rep_total = sum(r.token_stats[tok]["repr_tokens"] for r in parsed if tok in r.token_stats)
                saved = py_total - rep_total
                saved_pct = (saved / py_total) * 100 if py_total else 0.0
                rows.append(
                    {
                        "representation": rep,
                        "dataset": ds,
                        "tokenizer": tok,
                        "total_cases": total,
                        "parse_ok_cases": parsed_n,
                        "functional_total_cases": functional_total,
                        "functional_ok_cases": functional_n,
                        "python_tokens": py_total,
                        "repr_tokens": rep_total,
                        "saved_tokens": saved,
                        "saved_pct": round(saved_pct, 2),
                    }
                )
    return rows


# ------------------------------ main -----------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-to-head representation benchmark harness")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval", "mbpp_train"],
        choices=["humaneval", "mbpp_train"],
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["cl100k_base", "o200k_base", "llama_tinyllama", "codegen_350m_mono"],
        choices=["cl100k_base", "o200k_base", "llama_tinyllama", "codegen_350m_mono"],
    )
    parser.add_argument(
        "--external-config",
        default=None,
        help="Path to JSON external adapter config",
    )
    parser.add_argument(
        "--include-simpy",
        action="store_true",
        help="Include SimPy baseline adapter",
    )
    parser.add_argument(
        "--simpy-root",
        default="external/SimPy",
        help="Path to SimPy repo root (default: external/SimPy)",
    )
    parser.add_argument(
        "--include-token-sugar",
        action="store_true",
        help="Include Token Sugar baseline adapter",
    )
    parser.add_argument(
        "--token-sugar-root",
        default="external/TokenSugar",
        help="Path to TokenSugar repo root (default: external/TokenSugar)",
    )
    parser.add_argument(
        "--token-sugar-pattern-file",
        default="external/TokenSugar/mined_sugars.json",
        help="Path to Token Sugar pattern file",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Optional limit of samples per dataset (0 = all)",
    )
    parser.add_argument(
        "--skip-functional",
        action="store_true",
        help="Skip HumanEval functional checks",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="Timeout seconds for HumanEval check_correctness",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizers = build_tokenizers(args.tokenizers)
    adapters = [python_adapter(), kern_adapter()]
    if args.include_simpy:
        adapters.append(simpy_adapter(args.simpy_root))
    if args.include_token_sugar:
        adapters.append(
            token_sugar_adapter(args.token_sugar_root, args.token_sugar_pattern_file)
        )
    adapters.extend(load_external_adapters(args.external_config))

    inputs: list[tuple[str, str, dict]] = []
    if "humaneval" in args.datasets:
        rows = load_humaneval()
        inputs.extend(rows[: args.max_cases] if args.max_cases > 0 else rows)
    if "mbpp_train" in args.datasets:
        rows = load_mbpp_train()
        inputs.extend(rows[: args.max_cases] if args.max_cases > 0 else rows)

    results: list[CaseResult] = []
    total = len(inputs) * len(adapters)
    done = 0
    print(f"Running head-to-head on {len(inputs)} samples x {len(adapters)} reps = {total} evals")
    print("Representations:", ", ".join(a.name for a in adapters))

    for adapter in adapters:
        for dataset, task_id, payload in inputs:
            res = evaluate_case(
                adapter=adapter,
                dataset=dataset,
                task_id=task_id,
                payload=payload,
                tokenizers=tokenizers,
                run_functional=not args.skip_functional,
                timeout=args.timeout,
            )
            results.append(res)
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  progress: {done}/{total}")

    summary = summarize(results, tokenizers=list(tokenizers.keys()), datasets=args.datasets)

    # Console summary
    print("\nSummary:")
    for row in summary:
        if row["dataset"] != "overall":
            continue
        print(
            f"  {row['representation']:<12} | {row['tokenizer']:<18} | "
            f"parse {row['parse_ok_cases']}/{row['total_cases']} | "
            f"functional {row['functional_ok_cases']}/{row['functional_total_cases']} | "
            f"{row['python_tokens']} -> {row['repr_tokens']} ({row['saved_pct']}%)"
        )

    failures = [
        r
        for r in results
        if (not r.parse_ok) or (r.functional_applicable and not r.functional_ok)
    ]
    if failures:
        print("\nFirst failures (max 20):")
        for r in failures[:20]:
            print(
                f"  - {r.representation}:{r.dataset}:{r.task_id} "
                f"stage={r.error_stage} msg={r.error_message}"
            )

    details_path = Path(__file__).with_name("head_to_head_details.json")
    summary_json = Path(__file__).with_name("head_to_head_summary.json")
    summary_csv = Path(__file__).with_name("head_to_head_summary.csv")

    details_payload = []
    for r in results:
        raw = asdict(r)
        details_payload.append(raw)

    details_path.write_text(json.dumps(details_payload, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "representation",
                "dataset",
                "tokenizer",
                "total_cases",
                "parse_ok_cases",
                "functional_total_cases",
                "functional_ok_cases",
                "python_tokens",
                "repr_tokens",
                "saved_tokens",
                "saved_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nArtifacts:")
    print(f"  {details_path}")
    print(f"  {summary_json}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
