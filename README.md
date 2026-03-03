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

Generated benchmark artifacts:
- `humaneval_roundtrip_report.json`
- `humaneval_functional_report.json`
- `benchmark_multitokenizer_summary.csv`
- `benchmark_multitokenizer_summary.json`
- `benchmark_multitokenizer_details.json`
- `head_to_head_summary.csv`
- `head_to_head_summary.json`
- `head_to_head_details.json`

## Quickstart

Install dependencies:

```bash
python3 -m pip install tiktoken human-eval datasets transformers sentencepiece
```

Run tests:

```bash
python3 test_transpiler.py
python3 test_roundtrip_full.py
```

Run benchmarks:

```bash
python3 benchmark_grammar.py
python3 benchmark_humaneval_roundtrip.py
python3 benchmark_humaneval_functional.py
python3 benchmark_multitokenizer.py
python3 benchmark_head_to_head.py --datasets humaneval mbpp_train --tokenizers cl100k_base
```

Run with external baseline adapters (SimPy / Token Sugar) once you have converters:

```bash
python3 benchmark_head_to_head.py \
  --datasets humaneval mbpp_train \
  --tokenizers cl100k_base o200k_base \
  --external-config head_to_head_external_example.json
```

## Notes

- `llama_tinyllama` is used as a practical tokenizer proxy for LLaMA-family tokenization.
- Benchmark scripts validate conversion before counting tokens (transpile, compile, and parse-back checks).
- `head_to_head_external_example.json` is a template; replace command paths with your real converters.
