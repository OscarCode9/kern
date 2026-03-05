# Compiler Guide (Kern -> Python)

## Scope

`kern_compiler.py` compiles Kern code back into readable, valid Python.

## CLI

```bash
python3 kern_compiler.py input.kern
cat input.kern | python3 kern_compiler.py
```

## Core Behavior

1. Lexes Kern tokens.
2. Parses statements and expressions with block-aware rules.
3. Reconstructs Python indentation and block structure.
4. Outputs readable Python source.

## Block Rule

Kern blocks use `{ ... }`.
Compiler converts blocks into `:\n` + proper Python indentation.
Empty block becomes `pass`.

## Keyword Mapping (high level)

- `fn` -> `def`
- `ret` -> `return`
- `cls` -> `class`
- `imp` -> `import`
- `exc` -> `except`
- `fin` -> `finally`
- `&&` -> `and`
- `||` -> `or`

Compatibility aliases accepted in expressions:

- `band` -> `&`
- `bor` -> `|`
- `bxor` -> `^`
- `yld` -> `yield`

## Supported Statement Families

- Function/class definitions
- Conditionals and loops
- Try/except/finally
- with / async with
- async def / async for
- raise / del / assert
- pass / break / continue
- global / nonlocal
- yield / yield from

## Parser Stability Notes

- `fn` is treated as function keyword only in function-definition context.
- `cls` is treated as class keyword only in class-definition context.
- `ret` can still be used as identifier where syntactically valid.
