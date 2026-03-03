"""
Buscar optimizaciones adicionales para superar 20%+
Testear: operadores lógicos, comentarios, docstrings, patrones comunes
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def tok(s):
    tokens = enc.encode(s)
    return len(tokens), tokens

def cmp(label, a, b):
    ta, _ = tok(a)
    tb, _ = tok(b)
    delta = ta - tb
    pct = (delta / ta * 100) if ta > 0 else 0
    win = "✓ MEJOR" if delta > 0 else ("= igual" if delta == 0 else "✗ PEOR")
    print(f"  {label:<30} {ta:>4} → {tb:>4}  ({delta:+d})  {win}")
    return ta, tb

print("=" * 65)
print("OPTIMIZACIÓN DE GRAMÁTICA — búsqueda de ganancias adicionales")
print("=" * 65)

print("\n── 1. OPERADORES LÓGICOS ──────────────────────────────────")
cmp("'and' vs '&'",      "and",           "&")
cmp("'or' vs '|'",       "or",            "|")
cmp("'not' vs '!'",      "not",           "!")
cmp("x > 0 and x < 10",  "x > 0 and x < 10",  "x>0&x<10")
cmp("x < 0 or x > 10",   "x < 0 or x > 10",   "x<0|x>10")
cmp("not x vs !x",        "not x",              "!x")
cmp("if not lst vs if!lst","if not lst:",       "if!lst{")

print("\n── 2. DOCSTRINGS (frecuentes en código real) ─────────────")
cmp("docstring corto",
    '"""Add two numbers."""',
    "")  # strip = 0 tokens
cmp("docstring mediano",
    '"""Find the maximum value in a list."""',
    "")
cmp("docstring largo",
    '"""Calculate the fibonacci sequence up to n terms."""',
    "")

print("\n── 3. COMENTARIOS INLINE ──────────────────────────────────")
cmp("# comment corto",   "# add numbers",    "")
cmp("# comment mediano", "# check if empty", "")

print("\n── 4. TRUE/FALSE/NONE ─────────────────────────────────────")
cmp("'True' vs 'T'",   "True",  "T")
cmp("'False' vs 'F'",  "False", "F")
cmp("'None' vs 'N'",   "None",  "N")
cmp("while True vs wT", "while True{", "wT{")  # bad idea but testing

print("\n── 5. SELF EN MÉTODOS ─────────────────────────────────────")
cmp("'self' vs 's'",    "self",       "s")
cmp("self.x vs s.x",    "self.x",     "s.x")
cmp("self.val vs s.v",  "self.val",   "s.val")
cmp("fn __init__(self", "def __init__(self,", "fn __init__(s,")

print("\n── 6. NONE/PASS/RAISE ─────────────────────────────────────")
cmp("'raise' vs 'rai'", "raise", "rai")
cmp("'pass' vs '_'",    "pass",  "_")

print("\n── 7. OPERADORES CON FUSIÓN ───────────────────────────────")
# Key insight: removing spaces can cause BPE fusion
cmp("a + b vs a+b",      "a + b",   "a+b")
cmp("x == 0 vs x==0",    "x == 0",  "x==0")
cmp("x != 0 vs x!=0",    "x != 0",  "x!=0")
cmp("x >= 0 vs x>=0",    "x >= 0",  "x>=0")
cmp("a and b vs a&b",    "a and b", "a&b")
cmp("a or b vs a|b",     "a or b",  "a|b")

print("\n── 8. REALISTA: función con docstring+comentarios ─────────")
py_with_docs = '''def process_items(items, threshold=0.5):
    """Process a list of items filtering by threshold.

    Args:
        items: list of numeric values
        threshold: cutoff value
    Returns:
        filtered list
    """
    # filter items above threshold
    result = []
    for item in items:
        if item > threshold:  # keep only high values
            result.append(item)
    return result'''

kern_with_docs = "fn process_items(items,threshold=0.5){result=[];for item in items{if item>threshold{result.append(item)}};ret result}"
kern_no_docs   = kern_with_docs  # docstrings stripped automatically

ta, _ = tok(py_with_docs)
tb, _ = tok(kern_with_docs)
print(f"  {'Python (con docs):':<30} {ta:>4} tokens")
print(f"  {'Kern (sin docs):':<30} {tb:>4} tokens  ({ta-tb:+d})  {(ta-tb)/ta*100:.1f}%")

print("\n── 9. BOOLEAN OPS DENTRO DE IF ───────────────────────────")
cmp("if x > 0 and y > 0:",
    "if x > 0 and y > 0:",
    "if x>0&y>0{")
cmp("if not found or retry:",
    "if not found or retry:",
    "if!found|retry{")

print("\n── 10. COMMON PATTERNS ────────────────────────────────────")
cmp("append pattern",
    "result.append(item)",
    "result.append(item)")  # same - just testing
cmp("len check",
    "if len(lst) == 0:",
    "if len(lst)==0{")
cmp("None check",
    "if x is None:",
    "if x is None{")  # 'is' stays, can't abbreviate

print("\n" + "=" * 65)
print("RESUMEN DE OPTIMIZACIONES RECOMENDADAS:")
print("=" * 65)
opts = [
    ("and → &",      *tok("x and y"),   *tok("x&y")),
    ("or → |",       *tok("x or y"),    *tok("x|y")),
    ("not → !",      *tok("not x"),     *tok("!x")),
    ("self → s",     *tok("self"),      *tok("s")),
    ("strip comments", 5, None,         0, None),
    ("strip docstrings", 12, None,      0, None),
]

total_old = total_new = 0
for label, told, _, tnew, _ in opts:
    if tnew is not None:
        delta = told - tnew
        print(f"  {label:<20} {told:>4} → {tnew:>4}  ({delta:+d})")
        total_old += told
        total_new += tnew

pct = (total_old - total_new) / total_old * 100 if total_old > 0 else 0
