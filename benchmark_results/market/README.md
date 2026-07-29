# Kern world-market benchmark — July 29, 2026

This is Kern's first directly reproduced language-level market comparison. It
does **not** claim that Kern has already beaten every language. It establishes
one audited result and an explicit queue for the remaining contenders.

## Reproduced result

Kern v0.4 compact beats Sigil 0.1.0 under the shared production-tokenizer
protocol while preserving far more executable behavior.

All representations receive the same code-only canonical solutions:

- HumanEval+: 164;
- MBPP+: 378;
- BigCodeBench v0.1.4: 1,140;
- total: 1,682 programs.

The harness counts every encoded result, even when it cannot decode or execute.
Decode failures therefore cannot disappear from the token denominator.

### Token density

`cl100k_base` totals:

| Dataset | Python | Kern compact | python-minifier | Sigil 0.1.0 |
|---|---:|---:|---:|---:|
| HumanEval+ | `10,571` | **`7,342`** | `7,813` | `11,008` |
| MBPP+ | `15,183` | **`10,459`** | `11,565` | `13,130` |
| BigCodeBench | `163,786` | **`118,401`** | `123,107` | `149,382` |
| **Combined** | **`189,540`** | **`136,202`** | **`142,485`** | **`173,520`** |
| **Saved vs Python** | — | **`28.14%`** | **`24.83%`** | **`8.45%`** |

Kern compact uses `37,318` fewer `cl100k_base` tokens than Sigil (`21.51%`
below Sigil output). With `o200k_base`, Kern uses `139,556` tokens versus
Sigil's `176,043`, a difference of `36,487` (`20.73%` below Sigil).

Sigil expands HumanEval+ by `4.13%` under `cl100k_base`; its positive aggregate
result comes from `13.52%` savings on MBPP+ and `8.79%` on BigCodeBench.

![Shared-tokenizer market comparison](market-token-efficiency.svg)

### Structural coverage

| Representation | Fully converted | Decoded | Parseable Python | Contract AST |
|---|---:|---:|---:|---:|
| Kern compact | `1,682/1,682` | `1,682/1,682` | **`1,670/1,682`** | **`1,657/1,682`** |
| python-minifier | `1,682/1,682` | `1,682/1,682` | **`1,682/1,682`** | Diagnostic only |
| Sigil 0.1.0 | `892/1,682` | `282/1,682` | **`224/1,682`** | `0/1,682` strict AST |

Sigil's strict AST column is diagnostic, not its declared contract. Official
functional execution below is the stronger semantic evidence.

![Decoded structural coverage](market-structural-coverage.svg)

### Official EvalPlus execution

| Dataset | Python | Kern compact | python-minifier | Sigil 0.1.0 |
|---|---:|---:|---:|---:|
| HumanEval+ | `163/164` | **`163/164`** | `163/164` | `1/164` |
| MBPP+ | `378/378` | **`378/378`** | `378/378` | `3/378` |
| **Combined** | **`541/542`** | **`541/542`** | **`541/542`** | **`4/542`** |

Kern matches the Python reference outcome on all `542/542` tasks. The common
`HumanEval/32` numeric-oracle failure remains unchanged.

![EvalPlus market preservation](market-evalplus-correctness.svg)

## What this proves

The evidence supports this bounded statement:

> On the pinned 1,682-program corpus with `cl100k_base` and `o200k_base`, Kern
> v0.4 compact is denser and substantially more reliable than the public Sigil
> 0.1.0 Python conversion pipeline.

It does not yet prove superiority over Toke's native tokenizer, KARN's
unreproduced headline, or model-generation systems.

## Remaining world-market gates

The machine-readable registry is
[`competitors.json`](competitors.json). Current priority:

1. Toke paired programs under shared `cl100k_base` and `o200k_base`;
2. a separate native-tokenizer contest: Toke BPE versus a trained Kern BPE;
3. KARN paired compile-and-run programs before accepting its 76% claim;
4. exact ShortCoder and Token Sugar method reproduction;
5. continued monitoring for new public languages, version changes, and
   third-party reproduction.

NURL is not an immediate production-tokenizer leader: its own reproducible
report says it requires a median roughly `1.7x` Python's tokens across eight
matched algorithms. AI Native Lang remains a workflow DSL and requires a
domain-specific comparison rather than the general Python corpus.

## Reproduce

```bash
python -m venv .venv-market
.venv-market/bin/pip install -r market-benchmark-requirements.txt
.venv-market/bin/python benchmark_market.py --run-functional --parallel 8
```

Sigil's public wheel builds its Tree-sitter parser on first use, so a C compiler
is required and the first run can spend several minutes compiling that parser.

Artifacts:

- `market-benchmark-summary.json`: metadata, aggregates, functional results,
  failure counts, and bounded examples;
- `market-benchmark-details.csv`: every case and every gate;
- `market-token-efficiency.svg`: shared-tokenizer density;
- `market-structural-coverage.svg`: decoded parse coverage;
- `market-evalplus-correctness.svg`: official functional preservation.
