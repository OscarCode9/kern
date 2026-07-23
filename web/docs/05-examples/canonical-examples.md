# Canonical Examples (10)

These examples define expected canonical style for documentation, testing, and agent training prompts.

## 1) Single-expression function

Python:
```python
def add(a, b):
    return a + b
```

Kern:
```kern
add(a,b)=a+b
```

## 2) Multi-statement function

Python:
```python
def clamp(x, lo, hi):
    v = max(lo, min(x, hi))
    return v
```

Kern:
```kern
clamp(x,lo,hi){>v=max(lo,min(x,hi))
```

## 3) If/else block

Python:
```python
def absval(x):
    if x >= 0:
        return x
    else:
        return -x
```

Kern:
```kern
absval(x){if x>=0:>x;else:>-x
```

## 4) For loop

Python:
```python
def sum_n(n):
    total = 0
    for i in range(n):
        total += i
    return total
```

Kern:
```kern
sum_n(n){total=0;for i in range(n):total+=i;>total
```

## 5) While loop

Python:
```python
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a
```

Kern:
```kern
gcd(a,b){while b!=0:a,b=b,a%b;>a
```

## 6) Try/except/finally

Python:
```python
def parse_int(s):
    try:
        return int(s)
    except ValueError:
        return 0
    finally:
        cleanup()
```

Kern:
```kern
parse_int(s){try:>int(s);exc ValueError:>0;fin:cleanup()
```

## 7) Class + method

Python:
```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
```

Kern:
```kern
cls Point{.__init__(x,y){.x=x;.y=y
```

## 8) Imports

Python:
```python
import os
from typing import List, Dict
```

Kern:
```kern
imp os
from typing imp List,Dict
```

## 9) Lambda

Python:
```python
f = lambda x, y: x + y
```

Kern:
```kern
f=\x,y:x+y
```

## 10) With statement

Python:
```python
def read_text(path):
    with open(path) as f:
        return f.read()
```

Kern:
```kern
read_text(path){with open(path) as f:>f.read()
```
