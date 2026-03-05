# Transpiler Guide (Python -> Kern)

## Scope

`kern_transpiler.py` converts Python source into compact Kern syntax.

## CLI

```bash
python3 kern_transpiler.py input.py
cat input.py | python3 kern_transpiler.py
```

## Core Behavior

1. Parses Python with `ast.parse`.
2. Emits deterministic Kern syntax.
3. Removes non-semantic string expression statements (docstrings/no-op string literals).
4. Compacts whitespace around operators and commas.

## Key Output Conventions

- `def` -> `fn`
- `return` -> `ret`
- `class` -> `cls`
- `import` -> `imp`
- `except` -> `exc`
- `finally` -> `fin`
- Lambda -> `\params:expr`
- Boolean ops -> `&&` and `||`

## Supported Statement Families

- Functions and async functions
- Class definitions
- If/elif/else
- For/while (with optional `else`)
- Try/except/else/finally
- Imports and `from ... import ...`
- With statements
- Raise / del / assert
- pass / break / continue
- global / nonlocal
- yield / yield from

## Determinism Contract

For the same Python input, the transpiler should produce the same Kern output.
This is critical for stable dataset generation and model training.

## Known Limits

- `match/case` is not yet implemented.
- Async `for` / `with` emission from transpiler is pending explicit node handlers.
