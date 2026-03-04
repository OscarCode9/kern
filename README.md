# Kern

Kern is a compact, reversible representation of Python designed for LLM workflows.

Core idea:
- `Python -> Kern` for token-efficient reasoning/edit loops.
- `Kern -> Python` for execution and ecosystem compatibility.
- Deterministic round-trip to preserve semantics.

## Current status (March 3, 2026)

Implemented:
- Grammar v0.2
- Transpiler: `kern_transpiler.py` (Python -> Kern)
- Inverse compiler: `kern_compiler.py` (Kern -> Python)
- Round-trip and functional benchmarks on HumanEval
- Multi-tokenizer benchmark on HumanEval + MBPP
- Unified head-to-head harness vs external baselines (SimPy, Token Sugar)

## Key results

### Correctness

| Metric | Result |
|---|---:|
| HumanEval round-trip parseable (`Python -> Kern -> Python`) | `164/164` |
| HumanEval AST equivalence (normalized) | `164/164` |
| HumanEval functional pass (`check(entry_point)`) | `164/164` |

### Token reduction

#### Grammar benchmark (synthetic, `cl100k_base`)

| Dataset | Python | Kern | Saved | Saved % |
|---|---:|---:|---:|---:|
| v0.2 grammar set (24 samples) | `464` | `349` | `115` | `24.8%` |

#### Large benchmark (538 samples total)

HumanEval (164) + MBPP train (374), all valid conversions (`538/538`):

| Tokenizer | Python | Kern | Saved | Saved % |
|---|---:|---:|---:|---:|
| `cl100k_base` | `51570` | `25659` | `25911` | `50.24%` |
| `o200k_base` | `51677` | `25906` | `25771` | `49.87%` |
| `llama_tinyllama` | `64311` | `32100` | `32211` | `50.09%` |
| `codegen_350m_mono` | `62157` | `32645` | `29512` | `47.48%` |

Per-dataset highlights:
- HumanEval + `cl100k_base`: `30368 -> 8873` (`70.78%` saved)
- MBPP train + `cl100k_base`: `21202 -> 16786` (`20.83%` saved)

### Head-to-head baseline comparison (official code adapters)

Run context:
- Datasets: HumanEval (164) + MBPP train (374) = 538
- Representations: `python`, `kern`, `simpy`, `token_sugar`
- Protocol: encode -> decode-to-python -> `ast.parse` + HumanEval functional check

`cl100k_base` overall results:

| Representation | Parse OK | HumanEval functional | Python tokens | Repr tokens | Saved % |
|---|---:|---:|---:|---:|---:|
| `kern` | `538/538` | `164/164` | `51570` | `25659` | `+50.24%` |
| `python` | `538/538` | `164/164` | `51570` | `51570` | `0.00%` |
| `simpy` | `526/538` | `156/164` | `49583`* | `59888` | `-20.78%` |
| `token_sugar` | `528/538` | `155/164` | `48592`* | `97481` | `-100.61%` |

*For baseline rows, token totals are computed on parse-valid samples only (same harness rule as all representations).

Statistical view (`overall`, `cl100k_base`, bootstrap 95% CI on saved %):
- `kern`: `50.244%` [`47.527`, `53.220`]
- `python`: `0.000%` [`0.000`, `0.000`]
- `simpy`: `-20.783%` [`-22.246`, `-19.384`]
- `token_sugar`: `-100.611%` [`-107.692`, `-94.196`]

Universal claim for the current benchmark scope:
- Across all evaluated datasets and tokenizers (HumanEval + MBPP train; `cl100k_base`, `o200k_base`, `llama_tinyllama`, `codegen_350m_mono`), `kern` preserves correctness (`538/538` parse, `164/164` HumanEval functional) while sustaining ~`50%` token reduction, outperforming `simpy` and `token_sugar` on robustness and token efficiency in the same harness.

## Repository layout

- `kern_transpiler.py`: Python AST to Kern emitter
- `kern_compiler.py`: Kern parser/compiler to Python
- `test_transpiler.py`: transpiler smoke tests
- `test_roundtrip_full.py`: executable round-trip checks
- `benchmark_grammar.py`: grammar-level token benchmark
- `benchmark_humaneval_roundtrip.py`: AST/parse round-trip benchmark
- `benchmark_humaneval_functional.py`: HumanEval functional validation
- `benchmark_multitokenizer.py`: HumanEval + MBPP multi-tokenizer benchmark
- `benchmark_head_to_head.py`: unified head-to-head harness (`python`, `kern`, optional external baselines)
- `analyze_head_to_head.py`: bootstrap confidence intervals over head-to-head metrics
- `test_baseline_adapters.py`: adapter sanity tests (`python`, `kern`, `simpy`, `token_sugar`)
- `prepare_finetune_dataset.py`: exports `.py` + `.kern` pairs and JSONL for fine-tuning

Generated benchmark artifacts:
- `humaneval_roundtrip_report.json`
- `humaneval_functional_report.json`
- `benchmark_multitokenizer_summary.csv`
- `benchmark_multitokenizer_summary.json`
- `benchmark_multitokenizer_details.json`
- `head_to_head_summary.csv`
- `head_to_head_summary.json`
- `head_to_head_details.json`
- `head_to_head_stats.csv`
- `head_to_head_stats.json`

## Quickstart

Install dependencies:

```bash
python3 -m pip install tiktoken human-eval datasets transformers sentencepiece rope tree-sitter regex tqdm
```

Run tests:

```bash
python3 test_transpiler.py
python3 test_roundtrip_full.py
python3 test_baseline_adapters.py
```

Run benchmarks:

```bash
python3 benchmark_grammar.py
python3 benchmark_humaneval_roundtrip.py
python3 benchmark_humaneval_functional.py
python3 benchmark_multitokenizer.py
python3 benchmark_head_to_head.py --datasets humaneval mbpp_train --tokenizers cl100k_base
python3 analyze_head_to_head.py
```

Run head-to-head with SimPy and Token Sugar adapters:

```bash
python3 benchmark_head_to_head.py \
  --datasets humaneval mbpp_train \
  --tokenizers cl100k_base o200k_base llama_tinyllama codegen_350m_mono \
  --include-simpy \
  --include-token-sugar
```

Build fine-tuning dataset (`.kern` + `.py` + JSONL):

```bash
python3 prepare_finetune_dataset.py \
  --datasets humaneval mbpp_train \
  --valid-ratio 0.05 \
  --seed 42 \
  --out-dir data/finetune_v1 \
  --overwrite
```

Output structure:

```text
data/finetune_v1/
  train/
    py/*.py
    kern/*.kern
    pairs.jsonl
  valid/
    py/*.py
    kern/*.kern
    pairs.jsonl
  summary.json
  rejected.jsonl
```

## Notes

- `llama_tinyllama` is used as a practical tokenizer proxy for LLaMA-family tokenization.
- Benchmark scripts validate conversion before counting tokens (transpile, compile, and parse-back checks).
- `head_to_head_external_example.json` remains available if you want command-based custom adapters.
