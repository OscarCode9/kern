"""
discover_token_efficiency.py

Analyze benchmark_multitokenizer_details.json and surface where Kern loses
relative token efficiency. The output is meant to drive grammar/transpiler
changes with data-backed priorities.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from datasets import load_dataset
from human_eval.data import read_problems


@dataclass
class CaseRow:
    dataset: str
    task_id: str
    python_tokens: int
    kern_tokens: int
    saved_tokens: int
    saved_pct: float
    source: str
    features: list[str]


FEATURE_ORDER = [
    "class_def",
    "try_stmt",
    "with_stmt",
    "lambda_expr",
    "list_comp",
    "dict_comp",
    "set_comp",
    "gen_expr",
    "fstring",
    "decorator",
    "type_hints",
    "bool_op",
    "chained_compare",
    "for_loop",
    "while_loop",
    "tuple_unpack_assign",
    "slice_use",
    "subscript_use",
    "aug_assign",
]


def load_source_map(datasets_needed: set[str]) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    if "humaneval" in datasets_needed:
        problems = read_problems()
        for task_id, p in problems.items():
            out[("humaneval", task_id)] = p["prompt"] + p["canonical_solution"]
    if "mbpp_train" in datasets_needed:
        ds = load_dataset("mbpp", split="train")
        for item in ds:
            out[("mbpp_train", f"MBPP/{item['task_id']}")] = item["code"]
    return out


def extract_features(source: str) -> list[str]:
    feats = set()
    try:
        tree = ast.parse(source)
    except Exception:
        return []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            feats.add("class_def")
            if node.decorator_list:
                feats.add("decorator")
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.decorator_list:
                feats.add("decorator")
            if node.returns is not None:
                feats.add("type_hints")
            for arg in (
                list(node.args.args)
                + list(node.args.posonlyargs)
                + list(node.args.kwonlyargs)
            ):
                if arg.annotation is not None:
                    feats.add("type_hints")
            if node.args.vararg and node.args.vararg.annotation is not None:
                feats.add("type_hints")
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                feats.add("type_hints")
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node.decorator_list:
                feats.add("decorator")
            self.generic_visit(node)

        def visit_Try(self, node: ast.Try) -> None:
            feats.add("try_stmt")
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            feats.add("with_stmt")
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            feats.add("with_stmt")
            self.generic_visit(node)

        def visit_Lambda(self, node: ast.Lambda) -> None:
            feats.add("lambda_expr")
            self.generic_visit(node)

        def visit_ListComp(self, node: ast.ListComp) -> None:
            feats.add("list_comp")
            self.generic_visit(node)

        def visit_DictComp(self, node: ast.DictComp) -> None:
            feats.add("dict_comp")
            self.generic_visit(node)

        def visit_SetComp(self, node: ast.SetComp) -> None:
            feats.add("set_comp")
            self.generic_visit(node)

        def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
            feats.add("gen_expr")
            self.generic_visit(node)

        def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
            feats.add("fstring")
            self.generic_visit(node)

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            feats.add("bool_op")
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if len(node.ops) > 1:
                feats.add("chained_compare")
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            feats.add("for_loop")
            self.generic_visit(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            feats.add("for_loop")
            self.generic_visit(node)

        def visit_While(self, node: ast.While) -> None:
            feats.add("while_loop")
            self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                if isinstance(t, (ast.Tuple, ast.List)):
                    feats.add("tuple_unpack_assign")
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            feats.add("aug_assign")
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            feats.add("subscript_use")
            if isinstance(node.slice, ast.Slice):
                feats.add("slice_use")
            self.generic_visit(node)

    Visitor().visit(tree)
    return [f for f in FEATURE_ORDER if f in feats]


def load_rows(details_path: Path, tokenizer: str) -> list[CaseRow]:
    raw = json.loads(details_path.read_text(encoding="utf-8"))
    datasets_needed = {row["dataset"] for row in raw}
    source_map = load_source_map(datasets_needed)

    out: list[CaseRow] = []
    for row in raw:
        if not (row.get("transpile_ok") and row.get("compile_ok") and row.get("parse_back_ok")):
            continue
        stats = row.get("token_stats", {}).get(tokenizer)
        if not stats:
            continue

        key = (row["dataset"], row["task_id"])
        source = source_map.get(key, "")
        feats = extract_features(source) if source else []

        out.append(
            CaseRow(
                dataset=row["dataset"],
                task_id=row["task_id"],
                python_tokens=stats["python_tokens"],
                kern_tokens=stats["kern_tokens"],
                saved_tokens=stats["saved_tokens"],
                saved_pct=float(stats["saved_pct"]),
                source=source,
                features=feats,
            )
        )
    return out


def summarize_features(rows: list[CaseRow]) -> list[dict]:
    overall = statistics.mean(r.saved_pct for r in rows) if rows else 0.0
    out: list[dict] = []
    for feat in FEATURE_ORDER:
        subset = [r for r in rows if feat in r.features]
        if not subset:
            continue
        mean_saved = statistics.mean(r.saved_pct for r in subset)
        median_saved = statistics.median(r.saved_pct for r in subset)
        out.append(
            {
                "feature": feat,
                "count": len(subset),
                "mean_saved_pct": mean_saved,
                "median_saved_pct": median_saved,
                "delta_vs_overall_pct": mean_saved - overall,
                "mean_python_tokens": statistics.mean(r.python_tokens for r in subset),
                "mean_kern_tokens": statistics.mean(r.kern_tokens for r in subset),
            }
        )
    return sorted(out, key=lambda x: x["delta_vs_overall_pct"])


def summarize_prevalence_lift(rows: list[CaseRow]) -> list[dict]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda r: r.saved_pct)
    q = max(1, len(rows) // 4)
    worst = sorted_rows[:q]

    out: list[dict] = []
    for feat in FEATURE_ORDER:
        overall_count = sum(1 for r in rows if feat in r.features)
        worst_count = sum(1 for r in worst if feat in r.features)
        if overall_count == 0:
            continue
        overall_rate = overall_count / len(rows)
        worst_rate = worst_count / len(worst)
        out.append(
            {
                "feature": feat,
                "overall_rate": overall_rate,
                "worst_quartile_rate": worst_rate,
                "lift": (worst_rate / overall_rate) if overall_rate > 0 else 0.0,
                "overall_count": overall_count,
                "worst_quartile_count": worst_count,
            }
        )
    return sorted(out, key=lambda x: x["lift"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover token efficiency bottlenecks.")
    parser.add_argument(
        "--details",
        default="benchmark_multitokenizer_details.json",
        help="Path to benchmark details JSON.",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        choices=["cl100k_base", "o200k_base", "llama_tinyllama", "codegen_350m_mono"],
        help="Tokenizer to analyze.",
    )
    parser.add_argument("--top-worst", type=int, default=20, help="How many worst cases to print/store.")
    parser.add_argument("--out-json", default="token_efficiency_discovery.json", help="Output JSON report.")
    args = parser.parse_args()

    rows = load_rows(Path(args.details), tokenizer=args.tokenizer)
    if not rows:
        raise SystemExit("No valid rows found for selected tokenizer.")

    overall_mean = statistics.mean(r.saved_pct for r in rows)
    overall_median = statistics.median(r.saved_pct for r in rows)

    worst_cases = sorted(rows, key=lambda r: r.saved_pct)[: args.top_worst]
    feature_summary = summarize_features(rows)
    prevalence_lift = summarize_prevalence_lift(rows)

    report = {
        "tokenizer": args.tokenizer,
        "total_valid_cases": len(rows),
        "overall": {
            "mean_saved_pct": overall_mean,
            "median_saved_pct": overall_median,
        },
        "worst_cases": [
            {
                "dataset": r.dataset,
                "task_id": r.task_id,
                "saved_pct": r.saved_pct,
                "python_tokens": r.python_tokens,
                "kern_tokens": r.kern_tokens,
                "features": r.features,
            }
            for r in worst_cases
        ],
        "feature_summary": feature_summary,
        "worst_quartile_prevalence_lift": prevalence_lift,
    }

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Valid cases: {len(rows)}")
    print(f"Overall mean saved:   {overall_mean:.2f}%")
    print(f"Overall median saved: {overall_median:.2f}%")
    print("")
    print("Worst cases (lowest saved_pct):")
    for i, r in enumerate(worst_cases[:10], start=1):
        print(
            f"  {i:>2}. {r.dataset}:{r.task_id:<18} "
            f"saved={r.saved_pct:>6.2f}% py={r.python_tokens:>4} ke={r.kern_tokens:>4} "
            f"features={','.join(r.features[:5]) or '-'}"
        )
    print("")
    print("Features hurting savings most (delta vs overall):")
    for row in feature_summary[:10]:
        print(
            f"  {row['feature']:<20} count={row['count']:>3} "
            f"mean={row['mean_saved_pct']:>6.2f}% "
            f"delta={row['delta_vs_overall_pct']:>+6.2f} pp"
        )
    print("")
    print("Overrepresented in worst quartile (lift):")
    for row in prevalence_lift[:10]:
        print(
            f"  {row['feature']:<20} lift={row['lift']:.2f}x "
            f"worst_rate={row['worst_quartile_rate']*100:>5.1f}% "
            f"overall_rate={row['overall_rate']*100:>5.1f}%"
        )
    print("")
    print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
