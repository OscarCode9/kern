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
fn add(a,b)=a+b
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
fn clamp(x,lo,hi){v=max(lo,min(x,hi));ret v}
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
fn absval(x){if x>=0{ret x}else{ret -x}}
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
fn sum_n(n){total=0;for i in range(n){total+=i};ret total}
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
fn gcd(a,b){while b!=0{a,b=b,a%b};ret a}
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
fn parse_int(s){try{ret int(s)}exc ValueError{ret 0}fin{cleanup()}}
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
cls Point{fn __init__(self,x,y){self.x=x;self.y=y}}
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
fn read_text(path){with open(path) as f{ret f.read()}}
```
