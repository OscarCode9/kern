"""
Kern Grammar v0.1 — Benchmark real contra cl100k_base
Validación de los 10 constructs principales
"""
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def tok(s):
    return len(enc.encode(s))

def compare(label, python, kern):
    p = tok(python)
    k = tok(kern)
    delta = p - k
    pct = (delta / p) * 100
    bar = "✓" if delta > 0 else ("✗" if delta < 0 else "=")
    return label, python, kern, p, k, delta, pct, bar

cases = [
    # 1. FUNCTION — single expression
    compare(
        "fn simple",
        "def add(a, b):\n    return a + b",
        "fn add(a,b)=a+b"
    ),
    # 1b. FUNCTION — multi-statement
    compare(
        "fn multi",
        "def clamp(x, lo, hi):\n    result = max(lo, min(x, hi))\n    return result",
        "fn clamp(x,lo,hi){result=max(lo,min(x,hi));ret result}"
    ),
    # 1c. FUNCTION — with type hints
    compare(
        "fn typed",
        "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "fn factorial(n:int)->int{if n<=1{ret 1};ret n*factorial(n-1)}"
    ),
    # 2. RETURN
    compare(
        "return expr",
        "return a + b",
        "ret a+b"
    ),
    # 3. IF/ELSE
    compare(
        "if/else",
        "if x > 0:\n    return x\nelse:\n    return -x",
        "if x>0{ret x}else{ret -x}"
    ),
    # 3b. IF/ELIF/ELSE
    compare(
        "if/elif/else",
        "if x > 0:\n    print('pos')\nelif x < 0:\n    print('neg')\nelse:\n    print('zero')",
        "if x>0{print('pos')}elif x<0{print('neg')}else{print('zero')}"
    ),
    # 4. FOR loop
    compare(
        "for loop",
        "for i in range(10):\n    print(i)",
        "for i in range(10){print(i)}"
    ),
    # 4b. FOR with tuple unpack
    compare(
        "for k,v",
        "for k, v in d.items():\n    total += v",
        "for k,v in d.items(){total+=v}"
    ),
    # 5. WHILE
    compare(
        "while",
        "while x > 0:\n    x -= 1",
        "while x>0{x-=1}"
    ),
    # 6. IMPORT
    compare(
        "import",
        "import numpy as np",
        "imp numpy as np"
    ),
    compare(
        "from import",
        "from os.path import join, exists",
        "from os.path imp join,exists"
    ),
    # 7. CLASS
    compare(
        "class simple",
        "class Dog(Animal):\n    sound = 'Woof'\n    def speak(self):\n        return self.sound",
        "cls Dog(Animal){sound='Woof';fn speak(self)=self.sound}"
    ),
    # 8. TRY/EXCEPT
    compare(
        "try/except",
        "try:\n    x = int(s)\nexcept ValueError:\n    x = 0",
        "try{x=int(s)}exc ValueError{x=0}"
    ),
    compare(
        "try/except/finally",
        "try:\n    f = open('data.txt')\nexcept FileNotFoundError as e:\n    log(e)\nfinally:\n    cleanup()",
        "try{f=open('data.txt')}exc FileNotFoundError as e{log(e)}fin{cleanup()}"
    ),
    # 9. LAMBDA
    compare(
        "lambda simple",
        "lambda x: x + 1",
        r"\x:x+1"
    ),
    compare(
        "lambda in sorted",
        "sorted(lst, key=lambda x: x[1])",
        r"sorted(lst,key=\x:x[1])"
    ),
    # 10. ASSIGN
    compare(
        "assign spaces",
        "result = a + b",
        "result=a+b"
    ),
    compare(
        "assign annotated",
        "x: int = 5",
        "x:int=5"
    ),
    # FULL FUNCTION — realistic (no docs)
    compare(
        "find_max (full)",
        "def find_max(numbers: list) -> int:\n    if not numbers:\n        raise ValueError('empty list')\n    result = numbers[0]\n    for n in numbers[1:]:\n        if n > result:\n            result = n\n    return result",
        "fn find_max(numbers:list)->int{if not numbers{raise ValueError('empty list')};result=numbers[0];for n in numbers[1:]{if n>result{result=n}};ret result}"
    ),
    # 11. LOGICAL OPS — and/or → &/|
    compare(
        "and → &",
        "x > 0 and x < 10",
        "x>0&x<10"
    ),
    compare(
        "or → |",
        "x < 0 or x > 10",
        "x<0|x>10"
    ),
    compare(
        "and in if",
        "if a > 0 and b > 0:",
        "if a>0&b>0{"
    ),
    # 12. DOCSTRING STRIP
    compare(
        "fn + short docstring",
        "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome.\"\"\"\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
        "fn is_palindrome(s:str)->bool{s=s.lower().replace(' ','');ret s==s[::-1]}"
    ),
    compare(
        "binary_search (with doc)",
        "def binary_search(arr, target):\n    \"\"\"\n    Binary search. Returns index or -1.\n    \"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "fn binary_search(arr,target){left,right=0,len(arr)-1;while left<=right{mid=(left+right)//2;if arr[mid]==target{ret mid}elif arr[mid]<target{left=mid+1}else{right=mid-1}};ret -1}"
    ),
]

# ── Print results ───────────────────────────────────────────────────
print(f"\n{'KERN GRAMMAR v0.1 — Token Benchmark':^70}")
print(f"{'Tokenizer: cl100k_base (GPT-4)':^70}\n")
print(f"{'Construct':<20} {'Python':>6} {'Kern':>6} {'Saved':>6} {'%':>7}  {''}")
print("─" * 60)

total_py = total_kern = 0
for label, py, kern, p, k, delta, pct, bar in cases:
    total_py += p
    total_kern += k
    print(f"{label:<20} {p:>6} {k:>6} {delta:>+6} {pct:>6.1f}%  {bar}")

print("─" * 60)
total_delta = total_py - total_kern
total_pct = (total_delta / total_py) * 100
print(f"{'TOTAL':<20} {total_py:>6} {total_kern:>6} {total_delta:>+6} {total_pct:>6.1f}%")

print(f"\n→ Reducción promedio: {total_pct:.1f}% sobre {len(cases)} ejemplos")
print(f"→ Objetivo del paper: >15-20%")
print(f"→ Status: {'✓ PASA el umbral' if total_pct >= 15 else '✗ NO pasa el umbral'}\n")

# ── Verify keyword assumptions ──────────────────────────────────────
print("VERIFICACIÓN DE KEYWORDS (deben ser 1 token cada uno):")
print("─" * 40)
keywords = {
    "fn": 1, "ret": 1, "cls": 1, "imp": 1,
    "exc": 1, "fin": 1, "while": 1, "whl": "EVITAR",
    "def": 1, "return": 1, "class": 1, "import": 1,
}
for kw, expected in keywords.items():
    actual = tok(kw)
    if expected == "EVITAR":
        status = f"⚠  {actual} tokens ← NO USAR" if actual > 1 else f"ok {actual} token"
    else:
        status = f"✓  {actual} token" if actual == expected else f"✗  {actual} tokens (esperado {expected})"
    print(f"  {kw:<10} {status}")
