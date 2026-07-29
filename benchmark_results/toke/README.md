# Kern vs Toke — public-pair audit, July 29, 2026

This benchmark adds Toke to Kern's reproduced market evidence without treating
Toke as a Python transpiler. The pinned
[toke-eval source](https://github.com/karwalski/toke-eval/tree/851f6d8b2cfedea22833f3787ad96c19e072e952)
publishes 1,000 generated Toke programs but only 60 matching Python reference
functions. Those 60 public pairs are the comparison denominator.

Each Python reference is converted into a standalone JSON-CLI program with the
same input/output convention used by the Toke solution. Kern compact and
python-minifier receive that exact Python source. Toke receives its paired
published source. No failed conversion or compiler check disappears from the
denominator.

## Shared-tokenizer result

| Tokenizer | Python | Kern compact | python-minifier | Toke |
|---|---:|---:|---:|---:|
| `cl100k_base` | `3,565` | **`3,012`** | `2,892` | `6,347` |
| `o200k_base` | `3,616` | **`3,066`** | `2,945` | `6,351` |

Under `cl100k_base`, Kern uses `3,335` fewer tokens than Toke: **`52.54%`
below the Toke source total**. Toke expands the equivalent Python total by
`78.04%`. Under `o200k_base`, Kern is `51.72%` below Toke and Toke expands
Python by `75.64%`.

The result is not carried by the truncated 922-token `task-a-0015` outlier.
Kern is smaller on `59/60` individual pairs; the median per-pair advantage is
`47.19%`. Removing that largest Toke source leaves `2,974` Kern tokens versus
`5,425` Toke tokens, still **`45.18%` below Toke**.

![Shared-tokenizer Toke comparison](toke-shared-tokenizer.svg)

This micro-corpus also exposes a real Kern target: python-minifier uses `120`
fewer `cl100k_base` tokens than Kern (`3.98%` below Kern). That does not erase
Kern's previously reproduced win over python-minifier on the diverse
1,682-program corpus; it shows that tiny typed functions remain a separate
optimization regime.

## Toke native-tokenizer observation

The official `toke-tokenizer==0.1.0` wheel uses:

- `3,906` tokens for the same 60 Toke sources;
- `6,347` tokens with `cl100k_base`;
- a reproduced native reduction of `38.46%` on this public subset.

Kern's `3,012` `cl100k_base` tokens are `22.89%` below Toke's `3,906` native
tokens on these pairs. This is a useful real-output observation, but it is
**not** the final native-tokenizer contest because Kern and Toke use different
vocabularies in this row. The separately reproduced equal-16K contest uses
`2,803` Kern tokens versus `3,906` Toke tokens and is documented in the
[native-tokenizer report](../native-tokenizer/README.md).

Toke's [PyPI package](https://pypi.org/project/toke-tokenizer/) describes its
headline `~52%` as BPE reduction versus `cl100k_base` **on Toke source**, not
as a 52% language-level win over Python. Toke's more detailed
[42-example table](https://github.com/karwalski/toke/blob/a3adcebddbdf4629b5289a6f317ac6678c6061c8/docs/reference/token-comparison.md)
reports `31%` fewer Toke-BPE tokens than Python-cl100k tokens and reports Toke
with the shared cl100k tokenizer as `80%` more expensive than Python.

## Structure and public smoke probe

| Gate | Kern compact | Toke |
|---|---:|---:|
| Representation produced | `60/60` | `60/60` published |
| Current compiler check | n/a | `29/60` |
| Decoded parse / contract AST | **`60/60`** | n/a |
| Deterministic probe matches Python | **`60/60`** | `29/60` |

Every Toke program that passes the current compiler also passes the single
public probe. The other 31 fail before execution:

| First current-compiler error | Programs |
|---|---:|
| `E2002` old `=` equality syntax | `25` |
| `E4070` immutable reassignment | `4` |
| `E2004` truncated/unclosed source | `1` |
| `E4031` type mismatch | `1` |

![Public functional smoke probe](toke-functional-probe.svg)

The probes are one Kern-authored deterministic valid-domain input per task.
They are not Toke's private held-out tests and are not presented as a
reproduction of Toke's private functional score.

## Audit of all 1,000 published Toke solutions

The pinned public corpus reproduces the official `87,903` cl100k token total.
Its current status is:

| Measure | Result |
|---|---:|
| Public Toke solutions | `1,000` |
| `cl100k_base` tokens | `87,903` |
| Toke native BPE tokens | `56,822` |
| Native reduction vs cl100k | `35.36%` |
| Pass current pinned compiler `--legacy --check` | `430/1,000` |
| Fail current pinned compiler | `570/1,000` |

The public Gate 1 CSV records `588` passes, `335` failures, and `77` missing
results (`588/923` among evaluated tasks). The current compiler changed
equality from `=` to `==` after these solutions were generated, so the
compiler and evaluation snapshots no longer reproduce that historical pass
count. The pinned
[Gate 2 report](https://github.com/karwalski/toke-eval/blob/851f6d8b2cfedea22833f3787ad96c19e072e952/data/eval_report_gate2.json)
also explicitly notes compiler-strictness drift and that its language-level
comparison used Toke function bodies against fuller reference
implementations.

## What this proves

The evidence supports this bounded statement:

> On the 60 public Toke/Python pairs rendered as equivalent JSON-CLI programs,
> Kern v0.4 compact is 52.54% smaller than the published Toke source under the
> same cl100k tokenizer, round-trips every pair, and matches the Python smoke
> oracle on all 60.

It does not prove superiority on Toke's private held-out tests, on the
unpublished exact 42-example source bundle, or under a Kern-native tokenizer
that has not yet been trained and held out correctly.

## Reproduce

```bash
git clone https://github.com/karwalski/toke.git /tmp/toke
git -C /tmp/toke checkout a3adcebddbdf4629b5289a6f317ac6678c6061c8
make -C /tmp/toke

git clone https://github.com/karwalski/toke-eval.git /tmp/toke-eval
git -C /tmp/toke-eval checkout 851f6d8b2cfedea22833f3787ad96c19e072e952

python -m venv .venv-market
.venv-market/bin/pip install -r market-benchmark-requirements.txt
.venv-market/bin/python benchmark_toke.py \
  --toke-eval /tmp/toke-eval \
  --toke-compiler /tmp/toke/toke \
  --run-functional
```

Pinned external artifacts:

- Toke compiler commit:
  `a3adcebddbdf4629b5289a6f317ac6678c6061c8`;
- toke-eval commit:
  `851f6d8b2cfedea22833f3787ad96c19e072e952`;
- `toke-tokenizer==0.1.0` universal wheel SHA256:
  `c33eee7501da85966f969dc0007afd30c9cc2a7f2a6fb5f8b769aa7762210fcb`.

Artifacts:

- `toke-public-pair-summary.json`: metadata, aggregates, native lane,
  structural gates, probes, and 1,000-source audit;
- `toke-public-pair-details.csv`: every pair, hash, token count, compiler gate,
  and probe result;
- `toke-shared-tokenizer.svg`: neutral language-density graph;
- `toke-functional-probe.svg`: bounded public smoke-probe graph.
