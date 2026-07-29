# Kern vs K, GolfScript, and J — executable token-density screen

This report closes the first adversarial compact-language gate. It compares
Kern with the three languages at the top of the
[code.golf all-hole byte ranking](https://code.golf/rankings/langs/all/all/bytes)
snapshot checked on July 29, 2026: K, GolfScript, and J.

The leaderboard is a discovery signal, not the benchmark result. Its leading
solutions are private and it measures UTF-8 bytes rather than LLM tokens. This
screen instead publishes fourteen complete matched programs, executes every
language, and counts all sources with the same production tokenizers.

## Result

Every implementation produced the expected normalized stdout on all `14/14`
programs. Kern also passed its compact-AST contract and exact native-tokenizer
round-trip on `14/14`.

### Shared `cl100k_base`

| Language | Tokens | Exact outputs |
|---|---:|---:|
| GolfScript | **`169`** | `14/14` |
| J | **`163`** | `14/14` |
| **Kern compact** | `200` | **`14/14`** |
| K | `206` | `14/14` |
| Python | `265` | `14/14` |
| python-minifier | `217` | `14/14` |

Kern is `2.91%` below K under the neutral shared tokenizer. GolfScript and J
remain smaller: Kern is `18.34%` above GolfScript and `22.70%` above J in this
lane.

### Deployable system lane

Kern has a pinned native tokenizer; the three competitors do not publish
native LLM tokenizers in the evaluated runtime artifacts. The system lane
therefore compares Kern + Kern-16K with each competitor + `cl100k_base` and is
reported separately from the neutral language lane.

| Deployable language/tokenizer system | Tokens | Exact outputs |
|---|---:|---:|
| **Kern compact + Kern-16K** | **`159`** | **`14/14`** |
| J + `cl100k_base` | `163` | `14/14` |
| GolfScript + `cl100k_base` | `169` | `14/14` |
| K + `cl100k_base` | `206` | `14/14` |

On this fixed aggregate, Kern is `2.45%` below J, `5.92%` below GolfScript,
and `22.82%` below K.

![Shared-tokenizer result](compact-language-token-density.svg)

![Native-system result](compact-language-native-system.svg)

![Functional result](compact-language-functional.svg)

## What changed in Kern

The first run exposed repeated structural costs rather than isolated literals.
Three general compact-profile rules were implemented before the final run:

- scalar output `print(value)` becomes `::value`;
- iterable output `print(*values)` becomes `$values`;
- exact reverse slices `value[::-1]` become postfix `value~`;
- constant positive modulo filters such as
  `(x for x in range(1, 21) if x % 2 == 0)` become
  `range(2, 21, 2)`.

The default reversible emitter is unchanged. The compact compiler reconstructs
ordinary Python, and the output markers are statement-only so they cannot
capture Python identifiers.

The broader 1,682-program modern structural benchmark also improved:

| Dataset | Before | After | Additional `cl100k_base` saving |
|---|---:|---:|---:|
| HumanEval+ | `7,342` | **`7,324`** | `18` |
| MBPP+ | `10,459` | **`10,451`** | `8` |
| BigCodeBench | `118,401` | **`118,395`** | `6` |

HumanEval+ and MBPP+ preserve their compact AST contracts on every task.
BigCodeBench keeps the same `1,115/1,140` contract count and gains one
parseable result (`1,129/1,140` after versus `1,128/1,140` before).

## Protocol

The corpus covers five deliberately visible categories:

- two scalar programs;
- two reductions;
- three text programs;
- six array programs;
- one recurrence.

For each pair, the harness:

1. executes the Python source as the output oracle;
2. emits and compiles Kern compact;
3. executes python-minifier, K, GolfScript, and J;
4. collapses display-only whitespace while preserving values and order;
5. requires exact normalized stdout;
6. counts every complete source under `cl100k_base` and `o200k_base`;
7. counts Kern separately under the held-out Kern-16K tokenizer;
8. publishes sources, SHA-256 hashes, runtime gates, details, and totals.

Pinned runtime evidence:

| Runtime | Pin / gate |
|---|---|
| code.golf discovery repository | `2c0fc35ca0f76a2a6c7faaf4d32f21244a359a95` |
| ngn/k | `717063f24921d5aff405a39cf7643efedb5bb365` |
| GolfScript | `6155e9f7860775be53bdc79c6e1c3b9308ebbfe5`; script SHA-256 `c3d9800af812146c0215a8a61aa5fee615ccdb1bed3a3ff5f64b8b4e0a28c25e` |
| J | `9.6.3`; runtime version is checked at execution |
| Kern-16K | SHA-256 `d570a49067b2fffca939924527b8daa3c0cc74e687b1ce7026c3193396e570ec` |

The K and J executable SHA-256 values are also recorded in
`compact-language-summary.json`; they are platform-specific and are therefore
not presented as cross-platform release hashes.

## Reproduce

Prepare the external runtimes:

```bash
git clone https://codeberg.org/ngn/k.git /tmp/kern-k
git -C /tmp/kern-k checkout 717063f24921d5aff405a39cf7643efedb5bb365
make -C /tmp/kern-k

curl -L \
  https://raw.githubusercontent.com/lynn/golfscript/6155e9f7860775be53bdc79c6e1c3b9308ebbfe5/golfscript.rb \
  -o /tmp/golfscript.rb
chmod +x /tmp/golfscript.rb
```

Install the official J `9.6.3` runtime for the host platform, then run:

```bash
uv run \
  --with python-minifier \
  --with tiktoken \
  --with tokenizers \
  python benchmark_compact_languages.py \
  --k-root /tmp/kern-k \
  --golfscript /tmp/golfscript.rb \
  --j-binary /path/to/j9.6/bin/jconsole \
  --ruby /usr/bin/ruby

uv run \
  --with python-minifier \
  --with tiktoken \
  --with tokenizers \
  python -m unittest -v test_compact_languages_benchmark.py
```

The GolfScript runner requires Ruby in binary source-encoding mode. The harness
supplies `--encoding ASCII-8BIT` and rejects a script whose SHA-256 does not
match the pin.

## Limits

This is a bounded executable screen, not proof that Kern is the smallest
language in the world.

- The fourteen competitor programs were written for this benchmark from
  documented primitives; they are compact but are not claimed to be globally
  minimal or authored by each language's best golfer.
- code.golf's leading sources are private, so their byte totals cannot be
  recounted as LLM tokens.
- The corpus is intentionally small and contains six array tasks, a favorable
  domain for K and J; category totals remain public.
- Kern wins the native aggregate, but only `6/14` individual programs against
  GolfScript and `6/14` against J. A larger expert-reviewed corpus is still a
  required gate.
- Kern loses the UTF-8 byte lane (`429` bytes versus K `286`, GolfScript `260`,
  and J `277`). Byte golf and production-token density are different metrics.
- No code-generation model is evaluated here.

The next density screens are Pyth, Jelly, Uiua, and BQN, followed by the direct
agent-language audit of zerolang. A world-champion claim remains blocked until
those gates and independent expert solutions are available.

## Artifacts

- `compact-language-summary.json`: metadata, runtime gates, aggregates, and
  category totals;
- `compact-language-corpus.json`: all matched sources and expected outputs;
- `compact-language-details.csv`: per-program hashes, tokens, and correctness;
- `compact-language-token-density.svg`: shared-tokenizer comparison;
- `compact-language-native-system.svg`: explicitly separated native-system
  comparison;
- `compact-language-functional.svg`: complete functional denominator.
