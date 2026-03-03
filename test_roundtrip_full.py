"""
test_roundtrip_full.py — Valida que Python → Kern → Python produce código ejecutable.
No verifica igualdad exacta de texto, sino que el Python resultante:
  1. Parsea sin error (ast.parse)
  2. Ejecuta correctamente
"""
import ast
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from kern_transpiler import transpile
from kern_compiler  import compile_kern

SEP = "=" * 65
TESTS = [
    ("add", """
def add(a, b):
    return a + b
assert add(2, 3) == 5
assert add(-1, 1) == 0
"""),
    ("factorial", """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
assert factorial(5) == 120
assert factorial(0) == 1
"""),
    ("is_palindrome", """
def is_palindrome(s):
    \"\"\"Check palindrome.\"\"\"
    s = s.lower().replace(' ', '')
    return s == s[::-1]
assert is_palindrome('racecar')
assert not is_palindrome('hello')
"""),
    ("binary_search", """
def binary_search(arr, target):
    \"\"\"Binary search — returns index or -1.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
assert binary_search([1,2,3,4,5], 3) == 2
assert binary_search([1,2,3,4,5], 6) == -1
"""),
    ("find_max", """
def find_max(numbers):
    \"\"\"Find max in list.\"\"\"
    if not numbers:
        raise ValueError('empty list')
    result = numbers[0]
    for n in numbers[1:]:
        if n > result:
            result = n
    return result
assert find_max([3, 1, 4, 1, 5, 9]) == 9
assert find_max([42]) == 42
"""),
    ("count_words", """
def count_words(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts
result = count_words('a b a c a b')
assert result['a'] == 3
assert result['b'] == 2
"""),
    ("lambda_sort", """
words = ['banana', 'apple', 'cherry']
result = sorted(words, key=lambda x: x[0])
assert result[0] == 'apple'
assert result[1] == 'banana'
"""),
    ("list_comprehension", """
squares = [x**2 for x in range(5) if x % 2 == 0]
assert squares == [0, 4, 16]
"""),
    ("class_basic", """
class Counter:
    def __init__(self, start=0):
        self.value = start
    def increment(self):
        self.value += 1
    def get(self):
        return self.value
c = Counter()
c.increment()
c.increment()
assert c.get() == 2
"""),
    ("try_except", """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
assert safe_divide(10, 2) == 5.0
assert safe_divide(1, 0) is None
"""),
    ("and_or", """
def classify(x):
    if x > 0 and x < 10:
        return 'small'
    elif x >= 10 or x < 0:
        return 'other'
    return 'zero'
assert classify(5) == 'small'
assert classify(15) == 'other'
assert classify(-1) == 'other'
"""),
    ("walrus", """
data = [1, 2, 3, 4, 5]
if n := len(data):
    total = n * 2
assert total == 10
"""),
    ("import_usage", """
from math import sqrt, floor
assert floor(sqrt(16)) == 4
"""),
]

print(SEP)
print("ROUND-TRIP: Python → Kern → Python (AST parse + execute)")
print(SEP)

ok = fail = 0
for name, py_src in TESTS:
    py_src = py_src.strip()
    try:
        kern = transpile(py_src)
        py_back = compile_kern(kern)

        # 1. Must parse as valid Python
        ast.parse(py_back)

        # 2. Must execute correctly (asserts pass)
        exec(compile(py_back, '<kern>', 'exec'))

        try:
            enc = __import__('tiktoken').get_encoding('cl100k_base')
            py_toks   = enc.encode(py_src)
            kern_toks = enc.encode(kern)
            pct = 100 * (1 - len(kern_toks) / len(py_toks))
            print(f"  [OK] {name:<25} {len(py_toks):>4} → {len(kern_toks):>4} tok  ({pct:+.1f}%)")
        except ImportError:
            print(f"  [OK] {name}")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        if 'kern' in dir():
            print(f"         KERN: {kern[:120]}")
        if 'py_back' in dir():
            print(f"         PY:   {py_back[:120]}")
        fail += 1

print(SEP)
print(f"Passed: {ok}/{ok+fail}")
