# Kern optimization discovery — July 23, 2026

This report consolidates three independent, read-only investigations into the
next token-compression opportunities for Kern. It covers the same 1,682
canonical programs used by the modern benchmark:

- HumanEval+: 164 programs;
- MBPP+: 378 programs;
- BigCodeBench canonical solutions: 1,140 programs.

These numbers are **prototype estimates, not released benchmark results**.
They identify candidates for the next implementation iteration. Every accepted
change must be remeasured by the repository benchmark and must preserve the
appropriate reversible or compact-mode contract.

## Implementation update — July 28, 2026

The first conservative compact-mode bundle is now implemented and has passed
the report's full acceptance gate. The validated measurements supersede the
prototype estimates without changing the historical baseline below:

| Result | `cl100k_base` | `o200k_base` |
|---|---:|---:|
| Previous Kern compact | `137,405` | `140,829` |
| Updated Kern compact | **`136,202`** | **`139,556`** |
| Realized saving | **`1,203`** | **`1,273`** |
| Advantage over python-minifier | **`6,283` (`4.41%`)** | **`5,887` (`4.05%`)** |

The implementation combines a tokenizer-friendly alias order with descendant
load collision protection, a wildcard guard for structural pattern matching,
guarded assign-return folding, bare returns, and terminal-branch flattening.
Official EvalPlus outcomes retain per-task parity with Python on all `542/542`
programs. Structural coverage is also unchanged: HumanEval+ `164/164`, MBPP+
`378/378`, and BigCodeBench `1,128/1,140` parse plus `1,115/1,140` compact-AST
matches. The aggressive and reversible-grammar candidates remain unimplemented
and separately scoped.

## Current baseline

Aggregate representation tokens:

| Representation | `cl100k_base` | `o200k_base` |
|---|---:|---:|
| Kern compact | `137,405` | `140,829` |
| python-minifier 3.2.0 | `142,485` | `145,443` |
| Current Kern advantage | `5,080` (`3.57%`) | `4,614` (`3.17%`) |

The Kern compact totals consist of:

| Corpus | `cl100k_base` | `o200k_base` |
|---|---:|---:|
| HumanEval+ | `7,398` | `7,518` |
| MBPP+ | `10,620` | `10,777` |
| BigCodeBench solutions | `119,387` | `122,534` |

## Compact-mode candidates

The strongest conservative compact-mode prototype combines BPE-aware aliases
with three control-flow simplifications:

| Candidate | Estimated saving `cl100k` | Estimated saving `o200k` | Contract |
|---|---:|---:|---|
| BPE-aware local aliases with profitability scoring | `688` | `793` | Compact-safe after collision guards |
| `x = expr; return x` → `return expr` | `347` | `356` | Conservative with data-flow guards |
| `return None` → `return` | `65` | `86` | Semantically equivalent |
| Flatten `else` after terminal `return`/`raise` | `64` | `80` | Conservative for terminal branches |
| **Combined conservative prototype** | **`1,143`** | **`1,285`** | Requires full validation |

Projected conservative totals:

| Corpus | Current cl/o | Prototype cl/o | Saving cl/o |
|---|---:|---:|---:|
| HumanEval+ | `7,398 / 7,518` | `7,331 / 7,439` | `67 / 79` |
| MBPP+ | `10,620 / 10,777` | `10,481 / 10,647` | `139 / 130` |
| BigCodeBench solutions | `119,387 / 122,534` | `118,450 / 121,458` | `937 / 1,076` |
| **Total** | **`137,405 / 140,829`** | **`136,262 / 139,544`** | **`1,143 / 1,285`** |

If reproduced after implementation, Kern's margin over python-minifier would
increase to:

- `6,223` fewer `cl100k_base` tokens (`4.37%` below minifier output);
- `5,899` fewer `o200k_base` tokens (`4.06%` below minifier output).

### Alias-order experiment

The current compact pass allocates aliases as `A`, `B`, `C`, and so on.
Nineteen alternative orders were profiled. The best experimental order,
starting with BPE-friendly lowercase and underscore forms, projected a larger
`828 / 867` token saving.

That larger estimate is not yet accepted. Before using lowercase or `_`
aliases, the scope builder must reserve free names used by descendants as well
as descendant aliases. Otherwise an alias in a parent scope could shadow a
global read in a nested scope. The implementation should therefore use an
exact profitability score plus free-variable collision analysis, then remeasure
the result.

## Reversible grammar candidates

These candidates preserve the original Python AST and belong to the default
reversible contract.

| Candidate syntax | Meaning | Estimated saving cl/o | Risk |
|---|---|---:|---|
| `x?y`, `x!y`, `x??y`, `x!!y` | `in`, `not in`, `is`, `is not` | `133 / 128` | Low-medium; contextual lexer rules |
| `{:x}` / `{:key=value}` | Identifier string keys in dicts | `88–330 / 100–317` | Low-medium; adaptive emission required |
| `source@target{...}` | Postfix `for target in source` | `425 / 396` | Medium; statement-only `@` context |
| `[expr|x@items?cond]` | Symbolic comprehension | `394 / 321` | Medium-high; nesting and precedence |
| `module::name@alias` | Compact `from module import name as alias` | `259 / 262` | Low-medium; statement-only grammar |
| Contextual `>alias` | Replace `as alias` in import/with/except | up to `190 / 0` | Use only when neither tokenizer regresses |
| `<expr` / `<<expr` | `yield expr` / `yield from expr` | Very small | Low frequency in the corpus |

The dictionary-key range comes from two independently tested syntaxes and
different profitability gates. The next iteration must select one unambiguous
grammar, implement it in both emitter and compiler, and report the measured
result rather than adding the two estimates.

The postfix loop and symbolic comprehension forms are promising, but they
reuse characters that already participate in Python expressions. They should
follow the lower-risk compact and dictionary/operator changes, with the old
syntax retained as a fallback whenever the new spelling does not save tokens.

## Aggressive candidates kept separate

| Candidate | Estimated saving cl/o | Why it is not conservative |
|---|---:|---|
| Strip annotations | `841 / 899` | Changes `__annotations__`, type-hint inspection, Pydantic/dataclass/decorator behavior |
| Rename module globals | `1,075 / 1,421` | Changes public API and reflective name lookup |
| Remove assertions | `20 / 20` | Changes runtime behavior |

Removing annotations on top of the conservative compact prototype projected
totals of `135,456 / 138,678`, but it must be exposed as an explicit aggressive
option rather than folded into normal compact mode. Global renaming is not
recommended for the current contract.

## Ideas rejected by measurement

- Applying python-minifier's local-renaming pass before Kern increased output
  by `1,555 / 1,604` tokens.
- Literal hoisting increased output by `711 / 714` tokens.
- The full python-minifier transformation pipeline before Kern increased output
  by `1,312 / 1,292` tokens.
- Literal spelling selected in isolation was unstable across tokenizer
  boundaries: it lost `18` cl100k tokens overall despite saving `264` o200k
  tokens in one prototype.
- Forced braces around continuation suites regressed the large corpus.
- New shorthands for `if`, `for`, `while`, and unary `not` did not beat their
  existing one-token keyword forms.
- Singleton-tuple changes had no meaningful corpus coverage.

## Acceptance gate for the next iteration

An optimization will be published as implemented only after it passes all of
the following:

1. exact AST comparison against the correct reversible or compact reference;
2. official EvalPlus execution with per-task parity against Python;
3. all 1,682 programs retained in the benchmark denominator;
4. measurement with both `cl100k_base` and `o200k_base`;
5. no new parse or AST failures on the modern corpus;
6. regression tests for closures, comprehensions, `nonlocal`, imports,
   reflection, exceptions, `try/finally`, and alias collisions;
7. separate reporting for conservative and aggressive contracts;
8. no addition of independent prototype estimates without a combined rerun.

## Recommended implementation order

1. BPE-aware alias scoring with descendant/free-name collision protection.
2. Compact-mode assign-return, empty return, and terminal-branch transforms.
3. Reversible identifier-dictionary shorthand.
4. Reversible contextual membership and identity operators.
5. Adaptive import aliases and from-import syntax.
6. Postfix loops and symbolic comprehensions only after parser stress tests.
7. Optional annotation stripping as a separately named aggressive profile.
