# LLM Contract for Kern

## Purpose

Define strict rules so agents can generate Kern reliably and with low ambiguity.

## Generation Priorities

1. Correctness over compression.
2. Deterministic style over stylistic variation.
3. Follow canonical syntax from `02-grammar/syntax.md`.

## Canonical Rules

### Rule: blocks

- Use `:stmt` when a suite has exactly one simple statement.
- Use `{...}` for empty, multi-statement, or nested-compound bodies.
- Separate adjacent simple statements with `;`.
- Omit `;` after a structural block close.
- At EOF, omit all remaining final statement-block closing braces.

### Rule: booleans

- Use `&&` for logical and.
- Use `||` for logical or.
- Use `not` for negation.

### Rule: functions

- Prefer `name(args)=expr` for single-expression return.
- Use `name(args){...}` for multi-statement bodies.
- Use `.name(args){...}` and `.attr` for an implicit plain `self`.
- Use `>expr` for return and `>x=expr` for assign-then-return.

### Rule: same-name keyword arguments

- Use `:x` only in a direct call argument slot to mean `x=x`.
- Keep `x=x` inside f-string expressions and lambda parameter defaults.

### Rule: null identity

- Use `value?` for `value is None`.
- Use `value!` for `value is not None`.
- Do not use these forms for equality (`==` / `!=`).

### Rule: imports

- Use `imp module`
- Use `from module imp name1,name2`

### Rule: spacing

- No spaces around assignment/operators where grammar allows compaction.
- No spaces after commas in compact forms.

## Output Quality Checks (agent-side)

Before returning output, agent should verify:

1. Expression brackets and braces are balanced.
2. Only the final chain of statement blocks may rely on EOF closure.
3. No mixed boolean style (`and` with `&&` in same expression style pass).
4. Functions and classes follow canonical v0.4 forms.

## Required Roundtrip Safety

For benchmark or dataset generation tasks, output must pass:

`Python -> Kern -> Python -> ast.parse`.

## Invalid Patterns to Avoid

- Using `&` and `|` when logical operators are intended.
- Emitting unsupported `match/case` constructs.
- Omitting separators between adjacent simple statements in `{}` blocks.

## Change Control

If grammar changes, update this file and `02-grammar/syntax.md` in the same commit.
