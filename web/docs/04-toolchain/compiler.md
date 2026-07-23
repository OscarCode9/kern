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

## Suite and EOF Rules

Kern blocks use `{ ... }`; a one-simple-statement suite may use `:stmt`.
Compiler converts blocks into `:\n` + proper Python indentation.
Empty block becomes `pass`.
At EOF, every still-open statement block closes implicitly. Explicit v0.2/v0.3
closing braces remain accepted.

## Keyword Mapping (high level)

- `name(...)` / legacy `fn name(...)` -> `def`
- `>` -> `return` (`ret` remains accepted as v0.2 input)
- `>x=expr` -> `x=expr` followed by `return x`
- `.method(...)` / legacy `fn .method(...)` -> `def method(self, ...)`
- leading `.attr` in an implicit-receiver function -> `self.attr`
- postfix `?` / `!` -> `is None` / `is not None`
- `:x` in a call argument slot -> `x=x`
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

- A bare `name(...)` is a definition only when followed by `=`, `->...=`, or
  a statement suite.
- Legacy `fn` is treated as a function keyword only in definition context.
- `cls` is treated as class keyword only in class-definition context.
- `ret` can still be used as identifier where syntactically valid.
- Braced compounds are self-delimiting; inline suites use `;` where a following
  sibling or continuation would otherwise be ambiguous.
- Unmatched or unclosed expression delimiters fail immediately with
  `SyntaxError`.
