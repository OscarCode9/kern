# KERN GRAMMAR SPEC v0.2 (synced with implementation - Mar 2026)
> Compact Python-transpilable language for LLMs · Token-first design

---

## Design Principles

1. **ASCII only** — no unicode outside string literals
2. **AST-reversible** — every Kern construct maps 1:1 to a Python AST node
3. **Token-first** — choices driven by BPE tokenizer behavior, not just character count
4. **Inline-first** — `{}` blocks eliminate newlines + indentation (the real savings)
5. **Comments stripped** — `#` comments and docstrings omitted (not in Python AST, round-trip safe)

### What actually saves tokens (validated against cl100k_base)

| Change | Example | Delta |
|--------|---------|-------|
| Remove space after `,` | `(a, b)` → `(a,b)` | -1 per comma |
| Remove space around operators | `a + b` → `a+b` | -1 per op |
| Inline block (no newline+indent) | `:\n    return x` → `=x` | -3 to -5 |
| Single-expr function | `def f(x):\n    return x+1` → `fn f(x)=x+1` | -5 total |
| Lambda `\` | `lambda x: x+1` → `\x:x+1` | -2 to -3 |
| `and` → `&&` (in context) | `x > 0 and x < 10` → `x>0&&x<10` | -2 |
| `or` → `\|\|` (in context) | `x < 0 or x > 10` → `x<0\|\|x>10` | -3 |
| Strip `#` comments | `# check empty` → *(omit)* | -3 to -5 each |
| Strip docstrings | `"""docstring"""` → *(omit)* | -5 to -15 each |

**Real-world impact (functions with docstrings):**
- Toy examples (no docs): **~18% reduction**
- Realistic code (with docs/comments): **~40% reduction**

### What does NOT save tokens

- Keyword abbreviation when both are 1 BPE token: `self`→`s`, `True`→`T`, `None`→`N` = 0 savings
- `while`→`whl` BREAKS BPE into `[wh][l]` = 2 tokens → WORSE, do not use
- Replacing boolean keywords alone (without surrounding-space compression) gives limited savings
- Changing indent depth (4-space = 2-space, both 1 token)

---

## Block Syntax

The central mechanism: `{}` replaces Python's `: + newline + indent` pattern.

```
# Python (multi-line only)
def f(x):
    y = x + 1
    return y

# Kern (can be inline OR multi-line)
fn f(x){y=x+1;ret y}
```

**Rule**: `{` immediately follows a block-introducing construct. `;` separates statements.
`{}` in expression context = dict/set literal (same as Python). Unambiguous because
blocks only follow: `fn`, `if`, `else`, `elif`, `while`, `for`, `cls`, `try`, `exc`, `fin`, `with`.

---

## Grammar Constructs (Core + Extended)

---

### 1. Function Definition: `def` → `fn`

```
fn NAME([PARAMS])[->TYPE] = EXPR      # single expression (implicit return)
fn NAME([PARAMS])[->TYPE] { STMTS }  # multi-statement body
async fn NAME([PARAMS]) { STMTS }    # async
```

**Params**: `NAME[:TYPE][=DEFAULT]` — no spaces after commas.

```python
# Python                              # Kern
def add(a, b):                        fn add(a,b)=a+b
    return a + b

def greet(name="World"):              fn greet(name="World")=f"Hello {name}"
    return f"Hello {name}"

def clamp(x, lo, hi):                 fn clamp(x,lo,hi){res=max(lo,min(x,hi));ret res}
    res = max(lo, min(x, hi))
    return res

def factorial(n: int) -> int:         fn factorial(n:int)->int{if n<=1{ret 1};ret n*factorial(n-1)}
    if n <= 1:
        return 1
    return n * factorial(n - 1)

async def fetch(url):                  async fn fetch(url){ret await session.get(url)}
    return await session.get(url)
```

**Token count** (first example):
- Python: `def`, ` add`, `(a`, `,`, ` b`, `):`, `\n`, `    `, `return`, ` a`, ` +`, ` b` = **12 tokens**
- Kern: `fn`, ` add`, `(a`, `,b`, `)=`, `a`, `+b` = **7 tokens** → **-42%**

---

### 2. Return: `return` → `ret`

```
ret [EXPR]
```

`return` and `ret` are both 1 BPE token — savings come from the surrounding
context (no newline+indent before `ret`, no space before its argument fusing).

```python
# Python          # Kern
return            ret
return x          ret x
return x + y      ret x+y
return x, y       ret x,y
```

---

### 3. Conditional: `if/elif/else` (keywords unchanged)

```
if EXPR { STMTS } [elif EXPR { STMTS }]* [else { STMTS }]
```

`if`, `elif`, `else` are already short — kept as-is. Savings come from removing
`:` and the newline+indent that follow them.

```python
# Python                              # Kern
if x > 0:                             if x>0{ret x}else{ret -x}
    return x
else:
    return -x

if x > 0:                             if x>0{print("pos")}elif x<0{print("neg")}else{print("zero")}
    print("pos")
elif x < 0:
    print("neg")
else:
    print("zero")
```

**Ternary**: Python's `x if c else y` is kept — already compact. Kern allows both.

**Token count** (if/else example, full body):
- Python: ~14 tokens (with newlines, indentation, colons)
- Kern: `if`, ` x`, `>`, `0`, `{`, `ret`, ` x`, `}`, `else`, `{`, `ret`, ` -`, `x`, `}` = **14... wait**

Savings are clearer in nested code where Kern avoids multiple indentation levels.

---

### 4. For loop: `for` (keyword unchanged)

```
for NAME [, NAME]* in EXPR { STMTS }
```

```python
# Python                              # Kern
for i in range(10):                   for i in range(10){print(i)}
    print(i)

for k, v in d.items():                for k,v in d.items(){total+=v}
    total += v
```

**List/dict comprehensions**: unchanged — already compact Python syntax.
```
[x*2 for x in lst if x>0]   # identical in Kern
{k:v for k,v in d.items()}
```

---

### 5. While loop: `while` (keyword unchanged — do NOT abbreviate)

```
while EXPR { STMTS }
```

> **Warning**: `whl` tokenizes as `[wh][l]` = 2 tokens in cl100k_base.
> `while` = 1 token. Keep the full keyword.

```python
# Python                              # Kern
while x > 0:                          while x>0{x-=1}
    x -= 1

while True:                           while True{data=queue.pop();process(data)}
    data = queue.pop()
    process(data)
```

---

### 6. Import: `import` → `imp`

```
imp MODULE [as ALIAS]
from MODULE imp NAME [, NAME]*
from MODULE imp *
```

`import` = 1 BPE token, `imp` = 1 BPE token — keyword savings are 0.
Savings come from: no spaces after commas in multi-import.

```python
# Python                              # Kern
import os                             imp os
import numpy as np                    imp numpy as np
from os.path import join, exists      from os.path imp join,exists
from typing import List, Dict         from typing imp List,Dict
from . import utils                   from . imp utils
```

---

### 7. Class: `class` → `cls`

```
cls NAME [(BASE [, BASE]*)] { STMTS }
```

Methods inside the body use the same `fn` syntax, separated by `;`.

```python
# Python                              # Kern
class Point:                          cls Point{fn __init__(self,x,y){self.x=x;self.y=y};fn __str__(self)=f"({self.x},{self.y})"}
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"({self.x},{self.y})"

class Dog(Animal):                    cls Dog(Animal){sound="Woof";fn speak(self)=self.sound}
    sound = "Woof"
    def speak(self):
        return self.sound
```

**Decorators**: unchanged — `@decorator` already compact.

---

### 8. Try/Except: `except` → `exc`, `finally` → `fin`

```
try { STMTS }
  exc [EXCTYPE [as NAME]] { STMTS }
  [exc [EXCTYPE] { STMTS }]*
  [else { STMTS }]
  [fin { STMTS }]
```

Multiple exception types use a tuple: `exc(TypeError,ValueError)`.

```python
# Python                              # Kern
try:                                  try{x=int(s)}exc ValueError{x=0}
    x = int(s)
except ValueError:
    x = 0

try:                                  try{f=open("data.txt")}exc FileNotFoundError as e{log(e)}fin{cleanup()}
    f = open("data.txt")
except FileNotFoundError as e:
    log(e)
finally:
    cleanup()

except (TypeError, ValueError):       exc(TypeError,ValueError){...}
```

---

### 9. Lambda: `lambda` → `\`

```
\[PARAM [, PARAM]*]:EXPR
```

`lambda ` (with trailing space) fuses into 1 token. `\x` fuses into 1 token for
short names. Net savings: **2-3 tokens per lambda**.

```python
# Python                              # Kern
lambda x: x + 1                       \x:x+1
lambda x, y: x + y                    \x,y:x+y
lambda: 42                            \:42
sorted(lst, key=lambda x: x[1])       sorted(lst,key=\x:x[1])
map(lambda s: s.strip(), lines)        map(\s:s.strip(),lines)
```

**Token count** (`lambda x: x + 1` vs `\x:x+1`):
- Python: `lambda`, ` x`, `:`, ` x`, ` +`, ` 1` = **6 tokens**
- Kern: `\x`, `:`, `x+`, `1` = **4 tokens** → **-33%**

**Parsing rule**: `\` at expression-start position always introduces a lambda.
`:` is the params/body separator. Inside the body expression, `:` only appears
in slices (`a[1:2]`) and dicts (`{k:v}`) where it's syntactically unambiguous
(inside `[]` or `{}`).

---

### 10. Assignment (whitespace rules)

Assignments use `=` as Python, but **no spaces** around operators.

```
NAME = EXPR              # simple
NAME : TYPE = EXPR       # annotated
NAME += EXPR             # augmented (all Python augmented ops)
NAME, NAME = EXPR        # tuple unpack
*NAME, NAME = EXPR       # starred unpack
```

```python
# Python                              # Kern
x = 5                                 x=5
x: int = 5                            x:int=5
total += value                        total+=value
a, b = b, a                           a,b=b,a
first, *rest = lst                    first,*rest=lst
```

**Token savings**: spaces around single-char operators save 1 token each.
`x = y` = 3 tokens, `x=y` = 2 tokens.

---

### 11. Logical operators: `and`/`or` → `&&`/`||` (canonical)

```
EXPR && EXPR     # and
EXPR || EXPR     # or
not EXPR         # not
```

**Important**: in Kern, boolean operators are emitted as `&&` / `||` to avoid
collision with bitwise operators. Savings still come mostly from removing spaces
around operators.

```python
# Python                              # Kern                    Savings
x > 0 and x < 10                      x>0&&x<10                 -2 tokens
x < 0 or x > 10                       x<0||x>10                 -3 tokens
if a > 0 and b > 0:                   if a>0&&b>0{              -2 tokens
x >= 0 and x <= 100                   x>=0&&x<=100              -2 tokens
```

**Bitwise stays bitwise**: `&`, `|`, `^` remain bitwise operators in Kern
expressions. For hand-written Kern, compiler also accepts name aliases
`band` / `bor` / `bxor` as compatibility input.

**`not`**: canonical form remains `not EXPR`.

---

### 12. Comments and docstrings: stripped on transpile

**Rule**: `#` comments and docstrings are **not emitted** in Kern output.

Rationale:
- `#` comments are **not in the Python AST** — round-trip is AST-preserving
- Docstrings are `Expr(Constant(...))` nodes — stripping them changes the AST,
  but they're excluded from functional correctness tests (HumanEval, MBPP)
- The LLM receives task context via natural language prompt, not docstrings
- Comments can be regenerated by the LLM when producing Python output

**Token savings** (measured):
```python
# Short docstring:   """Add two numbers."""         → 5 tokens stripped
# Medium docstring:  """Find max in a list."""       → 9 tokens stripped
# Long docstring:    """Binary search implementation...""" → 10+ tokens stripped
# Inline comment:    # check if empty               → 3-4 tokens stripped
```

**Impact on realistic code**: Python code on GitHub has ~15-25% of tokens in
comments/docstrings. Stripping them pushes Kern's reduction from ~18% (toy
examples) to **~40%** on real-world functions.

```python
# Python (with docstring):             39 tokens
def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome."""
    s = s.lower().replace(" ", "")
    return s == s[::-1]

# Kern (stripped):                     23 tokens → -41%
fn is_palindrome(s:str)->bool{s=s.lower().replace(" ","");ret s==s[::-1]}
```

---

## Ambiguity Resolution Reference

| Situation | Resolution rule |
|-----------|----------------|
| `{` after block keyword | Block (not dict) |
| `{` after `=`, `(`, `[`, `,`, `ret` | Dict/set literal |
| `\` at expression start | Lambda |
| `:` after `\...` params | Lambda body separator |
| `:` in `{k:v}` | Dict separator (inside `{}` = dict context) |
| `:` in `[a:b]` | Slice (inside `[]`) |
| `;` at statement level | Statement separator (NOT inside string) |

---

## Full Example: Round-trip

```python
# Python (original)
def find_max(numbers: list) -> int:
    if not numbers:
        raise ValueError("empty list")
    result = numbers[0]
    for n in numbers[1:]:
        if n > result:
            result = n
    return result
```

```
# Kern (compact)
fn find_max(numbers:list)->int{if not numbers{raise ValueError("empty list")};result=numbers[0];for n in numbers[1:]{if n>result{result=n}};ret result}
```

Token estimate:
- Python: ~38 tokens
- Kern: ~28 tokens → **-26%**

---

## Implementation Status (Compiler + Transpiler, Mar 2026)

| Python | Kern | Notes |
|--------|------|-------|
| `with X as y:` | `with X as y{...}` | implemented |
| `yield x` | `yield x` (`yld` accepted by compiler) | implemented |
| `yield from x` | `yield from x` (`yld from` accepted by compiler) | implemented |
| `@decorator` | `@decorator` | implemented |
| f-strings | `f"..."` | implemented |
| `async def` | `async fn` | implemented |
| `async for` | `async for` | compiler supports; transpiler `AsyncFor` pending |
| `async with` | `async with` | compiler supports; transpiler `AsyncWith` pending |
| `match/case` (3.10+) | TBD | not implemented |

Extended statements now implemented in both directions: `raise`, `del`,
`assert`, `pass`, `break`, `continue`, `global`, `nonlocal`.

---

## Validation Checklist (for grammar changes)

When changing grammar tokens, verify assumptions with tiktoken:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def tok(s):
    tokens = enc.encode(s)
    return len(tokens), tokens

# Verify keyword token counts
assert tok("fn")[0] == 1        # must be single token
assert tok("ret")[0] == 1       # must be single token
assert tok("cls")[0] == 1       # must be single token
assert tok("imp")[0] == 1       # must be single token
assert tok("exc")[0] == 1       # must be single token
assert tok("fin")[0] == 1       # must be single token
assert tok("whl")[0] != 1       # this one FAILS — do not use

# Verify operator space savings
assert tok("a+b")[0] < tok("a + b")[0]    # True: 2 < 3
assert tok("a,b")[0] < tok("a, b")[0]     # True: 2 < 3

# Verify lambda savings
assert tok(r"\x:x+1")[0] < tok("lambda x: x+1")[0]   # True: 4 < 6
```

---

*Spec synchronized with compiler/transpiler behavior (Mar 2026).*
