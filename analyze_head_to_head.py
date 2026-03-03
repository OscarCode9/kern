"""
analyze_head_to_head.py

Compute confidence intervals (bootstrap) for head-to-head metrics from
head_to_head_details.json.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MetricRow:
    representation: str
    dataset: str
    tokenizer: str
    total_cases: int
    parse_ok_cases: int
    parse_rate: float
    parse_rate_ci_low: float
    parse_rate_ci_high: float
    functional_total_cases: int
    functional_ok_cases: int
    functional_rate: float
    functional_rate_ci_low: float
    functional_rate_ci_high: float
    python_tokens: int
    repr_tokens: int
    saved_tokens: int
    saved_pct: float
    saved_pct_ci_low: float
    saved_pct_ci_high: float


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int((len(ys) - 1) * q)
    return ys[idx]


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rnd = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / len(sample))
    return quantile(boots, 0.025), quantile(boots, 0.975)


def bootstrap_ratio_ci(
    numerators: list[float],
    denominators: list[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    if not numerators or not denominators:
        return 0.0, 0.0
    assert len(numerators) == len(denominators)
    rnd = random.Random(seed)
    n = len(numerators)
    boots = []
    for _ in range(n_boot):
        idxs = [rnd.randrange(n) for _ in range(n)]
        num = sum(numerators[i] for i in idxs)
        den = sum(denominators[i] for i in idxs)
        boots.append((num / den) * 100 if den else 0.0)
    return quantile(boots, 0.025), quantile(boots, 0.975)


def main() -> None:
    details_path = Path("head_to_head_details.json")
    rows = json.loads(details_path.read_text(encoding="utf-8"))

    reps = sorted({r["representation"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    tokenizers = sorted(
        {
            tok
            for r in rows
            for tok in (r.get("token_stats") or {}).keys()
        }
    )

    out: list[MetricRow] = []
    for rep in reps:
        for ds in datasets + ["overall"]:
            if ds == "overall":
                subset = [r for r in rows if r["representation"] == rep]
            else:
                subset = [r for r in rows if r["representation"] == rep and r["dataset"] == ds]

            total_cases = len(subset)
            parse_binary = [1.0 if r["parse_ok"] else 0.0 for r in subset]
            parse_ok_cases = int(sum(parse_binary))
            parse_rate = (parse_ok_cases / total_cases * 100.0) if total_cases else 0.0
            parse_ci_low, parse_ci_high = bootstrap_ci(parse_binary)
            parse_ci_low *= 100.0
            parse_ci_high *= 100.0

            f_subset = [r for r in subset if r.get("functional_applicable", False)]
            functional_total_cases = len(f_subset)
            f_binary = [1.0 if r["functional_ok"] else 0.0 for r in f_subset]
            functional_ok_cases = int(sum(f_binary))
            functional_rate = (functional_ok_cases / functional_total_cases * 100.0) if functional_total_cases else 0.0
            if functional_total_cases:
                f_ci_low, f_ci_high = bootstrap_ci(f_binary)
                f_ci_low *= 100.0
                f_ci_high *= 100.0
            else:
                f_ci_low = f_ci_high = 0.0

            for tok in tokenizers:
                parsed_rows = [r for r in subset if r["parse_ok"] and tok in r.get("token_stats", {})]
                py_tokens = sum(r["token_stats"][tok]["python_tokens"] for r in parsed_rows)
                rep_tokens = sum(r["token_stats"][tok]["repr_tokens"] for r in parsed_rows)
                saved_tokens = py_tokens - rep_tokens
                saved_pct = (saved_tokens / py_tokens * 100.0) if py_tokens else 0.0

                per_case_saved = [
                    r["token_stats"][tok]["python_tokens"] - r["token_stats"][tok]["repr_tokens"]
                    for r in parsed_rows
                ]
                per_case_py = [r["token_stats"][tok]["python_tokens"] for r in parsed_rows]
                s_ci_low, s_ci_high = bootstrap_ratio_ci(per_case_saved, per_case_py)

                out.append(
                    MetricRow(
                        representation=rep,
                        dataset=ds,
                        tokenizer=tok,
                        total_cases=total_cases,
                        parse_ok_cases=parse_ok_cases,
                        parse_rate=round(parse_rate, 3),
                        parse_rate_ci_low=round(parse_ci_low, 3),
                        parse_rate_ci_high=round(parse_ci_high, 3),
                        functional_total_cases=functional_total_cases,
                        functional_ok_cases=functional_ok_cases,
                        functional_rate=round(functional_rate, 3),
                        functional_rate_ci_low=round(f_ci_low, 3),
                        functional_rate_ci_high=round(f_ci_high, 3),
                        python_tokens=py_tokens,
                        repr_tokens=rep_tokens,
                        saved_tokens=saved_tokens,
                        saved_pct=round(saved_pct, 3),
                        saved_pct_ci_low=round(s_ci_low, 3),
                        saved_pct_ci_high=round(s_ci_high, 3),
                    )
                )

    json_path = Path("head_to_head_stats.json")
    csv_path = Path("head_to_head_stats.csv")
    json_path.write_text(json.dumps([asdict(r) for r in out], indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(out[0]).keys()),
        )
        writer.writeheader()
        writer.writerows(asdict(r) for r in out)

    # Print short leaderboard view for overall + cl100k
    print("Overall leaderboard (cl100k_base):")
    rank = [r for r in out if r.dataset == "overall" and r.tokenizer == "cl100k_base"]
    rank = sorted(rank, key=lambda r: r.saved_pct, reverse=True)
    for r in rank:
        print(
            f"  {r.representation:<12} saved={r.saved_pct:>7.3f}% "
            f"[{r.saved_pct_ci_low:.3f}, {r.saved_pct_ci_high:.3f}] "
            f"parse={r.parse_rate:.2f}% functional={r.functional_rate:.2f}%"
        )
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
