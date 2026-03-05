# LLM Contract for Kern

## Purpose

Define strict rules so agents can generate Kern reliably and with low ambiguity.

## Generation Priorities

1. Correctness over compression.
2. Deterministic style over stylistic variation.
3. Follow canonical syntax from `02-grammar/syntax.md`.

## Canonical Rules

### Rule: blocks

- Use `{}` for block bodies.
- Separate statements inside a block with `;`.

### Rule: booleans

- Use `&&` for logical and.
- Use `||` for logical or.
- Use `not` for negation.

### Rule: functions

- Prefer `fn name(args)=expr` for single-expression return.
- Use `fn name(args){...}` for multi-statement bodies.

### Rule: imports

- Use `imp module`
- Use `from module imp name1,name2`

### Rule: spacing

- No spaces around assignment/operators where grammar allows compaction.
- No spaces after commas in compact forms.

## Output Quality Checks (agent-side)

Before returning output, agent should verify:

1. Brackets and braces balanced.
2. Every opened block is closed.
3. No mixed boolean style (`and` with `&&` in same expression style pass).
4. Functions and classes follow canonical keyword forms.

## Required Roundtrip Safety

For benchmark or dataset generation tasks, output must pass:

`Python -> Kern -> Python -> ast.parse`.

## Invalid Patterns to Avoid

- Using `&` and `|` when logical operators are intended.
- Emitting unsupported `match/case` constructs.
- Omitting separators between statements in `{}` blocks.

## Change Control

If grammar changes, update this file and `02-grammar/syntax.md` in the same commit.
