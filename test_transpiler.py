"""Test suite para el transpilador Kern v0.1"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kern_transpiler import transpile

SEP = "=" * 60

tests = [
    ("fn simple", "def add(a, b):\n    return a + b"),

    ("fn typed", "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),

    ("fn + docstring strip", 'def is_palindrome(s: str) -> bool:\n    """Check if a string is a palindrome."""\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]'),

    ("if/elif/else", 'if x > 0:\n    print("pos")\nelif x < 0:\n    print("neg")\nelse:\n    print("zero")'),

    ("for loop", "for k, v in d.items():\n    total += v"),

    ("while", "while x > 0:\n    x -= 1"),

    ("import", "import numpy as np\nfrom os.path import join, exists"),

    ("class", 'class Dog(Animal):\n    sound = "Woof"\n    def speak(self):\n        return self.sound'),

    ("try/except/finally", 'try:\n    f = open("data.txt")\nexcept FileNotFoundError as e:\n    log(e)\nfinally:\n    cleanup()'),

    ("lambda", "f = sorted(lst, key=lambda x: x[1])"),

    ("and/or → &/|", "if x > 0 and x < 10:\n    print(x)"),

    ("list comprehension", "result = [x**2 for x in range(10) if x % 2 == 0]"),

    ("dict comp", "d = {k: v for k, v in pairs if v > 0}"),

    ("walrus", "if n := len(data):\n    print(n)"),

    ("f-string", 'msg = f"Hello {name}, you are {age} years old"'),

    ("find_max (realistic)", ''.join([
        "def find_max(numbers: list) -> int:\n",
        '    """Find max in list."""\n',
        "    if not numbers:\n",
        '        raise ValueError("empty list")\n',
        "    result = numbers[0]\n",
        "    for n in numbers[1:]:\n",
        "        if n > result:\n",
        "            result = n\n",
        "    return result",
    ])),

    ("binary_search", ''.join([
        "def binary_search(arr, target):\n",
        '    """\n    Binary search.\n    Returns index or -1.\n    """\n',
        "    left, right = 0, len(arr) - 1\n",
        "    while left <= right:\n",
        "        mid = (left + right) // 2\n",
        "        if arr[mid] == target:\n",
        "            return mid\n",
        "        elif arr[mid] < target:\n",
        "            left = mid + 1\n",
        "        else:\n",
        "            right = mid - 1\n",
        "    return -1",
    ])),
]

print(SEP)
print("KERN TRANSPILER v0.1 — Test Suite")
print(SEP)

passed = failed = 0
for name, code in tests:
    try:
        kern = transpile(code)
        print(f"  [OK] {name}")
        print(f"       {kern}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1

print(SEP)
print(f"Passed: {passed}/{len(tests)}   Failed: {failed}")
