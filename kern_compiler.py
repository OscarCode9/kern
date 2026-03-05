"""
kern_compiler.py — Kern → Python
Compila código Kern de vuelta a Python legible con indentación correcta.
No requiere dependencias externas.
"""

import re
import sys

# ── Keyword maps ────────────────────────────────────────────────────
KERN_KW = {
    'fn': 'def', 'ret': 'return', 'cls': 'class', 'imp': 'import',
    'exc': 'except', 'fin': 'finally', 'yld': 'yield',
    'band': '&', 'bor': '|', 'bxor': '^',
}

# Inside expressions, only keep aliasing for bitwise symbols.
EXPR_NAME_MAP = {
    'band': '&',
    'bor': '|',
    'bxor': '^',
}

# ── Lexer ────────────────────────────────────────────────────────────
_TOK = re.compile(r"""
  (?P<FSTR>  f(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))  |  # f"..."
  (?P<STR>   (?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))   |  # "..." '...'
  (?P<NUM>   \d+(?:\.\d+)?(?:[eE][+-]?\d+)?)            |  # numbers
  (?P<OP>    &&|\|\||->>|->|:=|//|\*\*|<<|>>|
             \+=|-=|\*=|/=|//=|%=|\*\*=|\|=|&=|\^=|<<=|>>=|
             ==|!=|<=|>=|[+\-*/%@&|^~<>=!])              |  # operators
  (?P<SPEC>  [{}\[\]().,;:\\])                           |  # specials
  (?P<NAME>  [a-zA-Z_]\w*)                               |  # identifiers
  (?P<NL>    \n)                                         |  # newlines
  (?P<WS>    [ \t]+)                                     |  # whitespace
  (?P<UNK>   .)                                             # catch-all
""", re.VERBOSE)


class Token:
    __slots__ = ('t', 'v')
    def __init__(self, t, v): self.t = t; self.v = v
    def __repr__(self): return f'({self.t},{self.v!r})'


def _lex(src: str):
    toks = []
    for m in _TOK.finditer(src):
        k, v = m.lastgroup, m.group()
        if k == 'WS': continue
        if k == 'NL': toks.append(Token('NL', '\n'))
        elif k in ('FSTR', 'STR'): toks.append(Token('STR', v))
        elif k == 'NUM': toks.append(Token('NUM', v))
        elif k in ('OP', 'SPEC'): toks.append(Token('OP', v))
        elif k == 'NAME': toks.append(Token('NAME', v))
    toks.append(Token('EOF', ''))
    return toks


# ── Parser ───────────────────────────────────────────────────────────
class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos  = 0
        self.ind  = 0          # current indent level

    # ── Token helpers ──────────────────────────────────────────────
    @property
    def cur(self): return self.toks[self.pos]
    def peek(self, n=1): return self.toks[min(self.pos+n, len(self.toks)-1)]

    def eat(self, val=None):
        t = self.toks[self.pos]
        if val is not None and t.v != val:
            raise SyntaxError(f"Expected {val!r}, got {t.v!r} at pos {self.pos}")
        self.pos += 1
        return t

    def match(self, val):
        if self.cur.v == val:
            self.pos += 1
            return True
        return False

    def skip_nl(self):
        while self.cur.t == 'NL':
            self.pos += 1

    def _i(self): return '    ' * self.ind

    # ── Program ────────────────────────────────────────────────────
    def compile(self) -> str:
        lines = []
        self.skip_nl()
        while self.cur.t != 'EOF':
            s = self._stmt()
            if s: lines.append(s)
            self.skip_nl()
        return '\n'.join(lines)

    # ── Block: {stmts} → ":\n    stmt\n    stmt" ──────────────────
    def _block(self) -> str:
        self.eat('{')
        self.ind += 1
        ind = self._i()
        stmts = []
        self.skip_nl()
        while self.cur.v != '}' and self.cur.t != 'EOF':
            s = self._stmt()
            if s: stmts.append(ind + s)
            # skip ; and NL between stmts
            while self.cur.v == ';' or self.cur.t == 'NL':
                self.pos += 1
        self.eat('}')
        self.ind -= 1
        if not stmts:
            return ':\n' + self._i() + '    pass'
        return ':\n' + '\n'.join(stmts)

    # ── Statements ─────────────────────────────────────────────────
    def _stmt(self) -> str:
        c = self.cur
        v = c.v

        if c.t == 'NAME':
            # "fn" is a definition keyword only for: fn NAME(...)
            if v == 'fn' and self.peek().t == 'NAME' and self.peek(2).v == '(':
                return self._fn(False)
            # "cls" is a class keyword only for: cls NAME(...)|cls NAME{...}
            if (
                v == 'cls'
                and self.peek().t == 'NAME'
                and self.peek(2).v in {'(', '{'}
            ):
                return self._cls()
            if v == 'imp':   return self._import()
            if v == 'from':  return self._from()
            # "ret" is a keyword only when it is not being used as an identifier.
            if v == 'ret' and self.peek().v not in {
                '=', ':', ',', '.',
                '+=', '-=', '*=', '/=', '//=', '%=', '**=',
                '|=', '&=', '^=', '<<=', '>>=',
            }:
                return self._return()
            if v == 'if':    return self._if()
            if v == 'for':   return self._for()
            if v == 'while': return self._while()
            if v == 'try':   return self._try()
            if v == 'raise': return self._raise()
            if v == 'with':  return self._with()
            if v == 'del':   return self._del()
            if v == 'assert':return self._assert()
            if v == 'pass':  self.eat(); return 'pass'
            if v == 'break': self.eat(); return 'break'
            if v == 'continue': self.eat(); return 'continue'
            if v == 'global':   return self._names('global')
            if v == 'nonlocal': return self._names('nonlocal')
            if v == 'async': return self._async()
            if v == 'yld' or v == 'yield':   return self._yield()

        if c.t == 'OP' and v == '@':
            return self._decorated()

        return self._expr_stmt()

    def _fn(self, is_async: bool) -> str:
        self.eat('fn')
        name = self.eat().v    # function name
        self.eat('(')
        params = self._params()
        self.eat(')')
        ret_ann = ''
        if self.cur.v == '->':
            self.eat('->')
            ret_ann = ' -> ' + self._expr_until({'{', '='})
        prefix = 'async def' if is_async else 'def'
        if self.cur.v == '=':
            self.eat('=')
            body = self._expr_line()
            return f'{prefix} {name}({params}){ret_ann}:\n{self._i()}    return {body}'
        block = self._block()
        return f'{prefix} {name}({params}){ret_ann}{block}'

    def _cls(self) -> str:
        self.eat('cls')
        name = self.eat().v
        bases = ''
        if self.cur.v == '(':
            self.eat('(')
            bases = self._expr_list(')')
            self.eat(')')
        block = self._block()
        return f'class {name}({bases}){block}'

    def _import(self) -> str:
        self.eat('imp')
        names = self._csv_names()
        return 'import ' + names

    def _from(self) -> str:
        self.eat('from')
        mod = self._dotted_name()
        self.eat('imp')
        names = self._csv_names()
        return f'from {mod} import {names}'

    def _return(self) -> str:
        self.eat('ret')
        if self.cur.v in (';', '}') or self.cur.t in ('NL', 'EOF'):
            return 'return'
        return 'return ' + self._expr_line()

    def _if(self) -> str:
        self.eat('if')
        cond = self._expr_until({'{', ';'})
        block = self._block()
        s = 'if ' + cond + block
        while self.cur.v == 'elif':
            self.eat('elif')
            cond = self._expr_until({'{', ';'})
            s += '\n' + self._i() + 'elif ' + cond + self._block()
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._block()
        return s

    def _for(self) -> str:
        self.eat('for')
        target = self._expr_until({'in'})
        self.eat('in')
        iter_ = self._expr_until({'{', ';'})
        block = self._block()
        s = f'for {target} in {iter_}{block}'
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._block()
        return s

    def _while(self) -> str:
        self.eat('while')
        cond = self._expr_until({'{', ';'})
        block = self._block()
        s = 'while ' + cond + block
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._block()
        return s

    def _try(self) -> str:
        self.eat('try')
        s = 'try' + self._block()
        while self.cur.v == 'exc':
            self.eat('exc')
            exc_clause = ''
            # exc Type as name or exc(T1,T2) as name
            if self.cur.v == '(':
                self.eat('(')
                types = self._expr_list(')')
                self.eat(')')
                exc_clause = '(' + types + ')'
            elif self.cur.v not in ('{', ';') and self.cur.t != 'NL':
                exc_type = self._expr_until({'as', '{', ';'})
                exc_clause = ' ' + exc_type
            as_name = ''
            if self.cur.v == 'as':
                self.eat('as')
                as_name = ' as ' + self.eat().v
            s += '\n' + self._i() + 'except' + exc_clause + as_name + self._block()
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._block()
        if self.cur.v == 'fin':
            self.eat('fin')
            s += '\n' + self._i() + 'finally' + self._block()
        return s

    def _raise(self) -> str:
        self.eat('raise')
        if self.cur.v in (';', '}') or self.cur.t in ('NL', 'EOF'):
            return 'raise'
        exc = self._expr_until({'from', ';', '}'})
        if self.cur.v == 'from':
            self.eat('from')
            cause = self._expr_line()
            return f'raise {exc} from {cause}'
        return 'raise ' + exc

    def _with(self) -> str:
        self.eat('with')
        items = self._with_items()
        return 'with ' + items + self._block()

    def _with_items(self) -> str:
        parts = []
        while True:
            ctx = self._expr_until({'as', ',', '{'})
            if self.cur.v == 'as':
                self.eat('as')
                var = self._expr_until({',', '{'})
                parts.append(ctx + ' as ' + var)
            else:
                parts.append(ctx)
            if self.cur.v != ',':
                break
            self.eat(',')
        return ', '.join(parts)

    def _del(self) -> str:
        self.eat('del')
        return 'del ' + self._expr_line()

    def _assert(self) -> str:
        self.eat('assert')
        test = self._expr_until({',', ';', '}'})
        if self.cur.v == ',':
            self.eat(',')
            msg = self._expr_line()
            return f'assert {test}, {msg}'
        return 'assert ' + test

    def _names(self, kw: str) -> str:
        self.eat(kw)
        names = []
        names.append(self.eat().v)
        while self.cur.v == ',':
            self.eat(',')
            names.append(self.eat().v)
        return kw + ' ' + ', '.join(names)

    def _async(self) -> str:
        self.eat('async')
        if self.cur.v == 'fn': return 'async ' + self._fn(False)
        if self.cur.v == 'for': return 'async ' + self._for()
        if self.cur.v == 'with': return 'async ' + self._with()
        return 'async ' + self._stmt()

    def _yield(self) -> str:
        self.pos += 1  # eat 'yld' or 'yield'
        if self.cur.v == 'from':
            self.eat('from')
            return 'yield from ' + self._expr_line()
        if self.cur.v in (';', '}') or self.cur.t in ('NL', 'EOF'):
            return 'yield'
        return 'yield ' + self._expr_line()

    def _decorated(self) -> str:
        decorators = []
        while self.cur.v == '@':
            self.eat('@')
            decorators.append('@' + self._expr_line())
        body = self._stmt()
        return '\n'.join(decorators) + '\n' + self._i() + body

    def _expr_stmt(self) -> str:
        return self._expr_line()

    # ── Params (fn definitions) ────────────────────────────────────
    def _params(self) -> str:
        parts = []
        while self.cur.v != ')' and self.cur.t != 'EOF':
            if self.cur.v == '*' and self.peek().v == ',':
                self.eat('*'); parts.append('*')
                if self.cur.v == ',': self.eat(',')
                continue
            if self.cur.v == '*' and self.peek().t == 'NAME':
                self.eat('*')
                parts.append('*' + self._single_param())
                if self.cur.v == ',': self.eat(',')
                continue
            if self.cur.v == '**':
                self.eat('**')
                parts.append('**' + self._single_param())
                if self.cur.v == ',': self.eat(',')
                continue
            parts.append(self._single_param())
            if self.cur.v == ',': self.eat(',')
            else: break
        return ', '.join(parts)

    def _single_param(self) -> str:
        name = self.eat().v
        ann = ''
        default = ''
        if self.cur.v == ':':
            self.eat(':')
            ann = ': ' + self._expr_until({'=', ',', ')'})
        if self.cur.v == '=':
            self.eat('=')
            default = '=' + self._expr_until({',', ')'})
        return name + ann + default

    # ── Helper: names with "as" aliases ───────────────────────────
    def _csv_names(self) -> str:
        parts = []
        while True:
            name = self._dotted_name()
            if self.cur.v == 'as':
                self.eat('as')
                alias = self.eat().v
                parts.append(name + ' as ' + alias)
            else:
                parts.append(name)
            if self.cur.v != ',': break
            self.eat(',')
        return ', '.join(parts)

    def _dotted_name(self) -> str:
        # dots (relative imports) then name
        dots = ''
        while self.cur.v == '.':
            self.eat('.'); dots += '.'
        parts = []
        # Allow relative imports like: from . imp x
        # In that case, "imp" is the delimiter keyword, not module name.
        if self.cur.t == 'NAME' and not (dots and self.cur.v == 'imp'):
            parts.append(self.eat().v)
            while self.cur.v == '.' and self.peek().t == 'NAME' and self.peek().v != 'imp':
                self.eat('.'); parts.append(self.eat().v)
        return dots + '.'.join(parts)

    # ── Expression helpers ─────────────────────────────────────────
    def _expr_list(self, stop: str) -> str:
        parts = []
        while self.cur.v != stop and self.cur.t != 'EOF':
            parts.append(self._expr_until({',', stop}))
            if self.cur.v == ',': self.eat(',')
            else: break
        return ', '.join(parts)

    def _expr_until(self, stops: set) -> str:
        """Collect tokens until we hit a stop token (not inside brackets)."""
        parts = []
        depth = 0
        while self.cur.t != 'EOF':
            v = self.cur.v
            # Always check stops before depth tracking
            if depth == 0 and v in stops: break
            if depth == 0 and self.cur.t == 'NL': break
            if v in ('{', '(', '['):  depth += 1
            elif v in ('}', ')', ']'):
                if depth == 0: break   # unmatched close — stop
                depth -= 1
            # Lambda: \params:body — can appear at any depth
            if v == '\\':
                self.pos += 1
                parts.append(self._lambda_with_stops(stops))
                continue
            parts.append(self._next_tok())
        return ''.join(parts)

    def _lambda_with_stops(self, stops: set) -> str:
        r"""Parse \params:body inheriting the parent expression's stops."""
        # Collect raw param tokens until ':' at depth 0
        param_toks = []
        d = 0
        while self.cur.t != 'EOF':
            v = self.cur.v
            if v == ':' and d == 0: break
            if v in ('(', '[', '{'): d += 1
            elif v in (')', ']', '}'): d -= 1
            param_toks.append(v)
            self.pos += 1
        self.eat(':')
        # Join params, add space after comma for readability
        params_str = ''.join(param_toks).replace(',', ', ')
        # Parse body with same stops as parent
        body = self._expr_until(stops)
        return 'lambda ' + params_str + ': ' + body

    def _expr_line(self) -> str:
        # Trim boundary whitespace to avoid accidental over-indentation
        # when expressions start with spaced keywords like "await" or "not".
        return self._expr_until({';', '}'}).strip()

    # Keywords that need surrounding spaces when used inside expressions
    _SPACED_EXPR_KW = {
        'not', 'in', 'is', 'if', 'else', 'for', 'from', 'as', 'await', 'and', 'or',
        'yield',
    }

    def _next_tok(self) -> str:
        """Consume one token and translate to Python."""
        t = self.cur
        self.pos += 1
        if t.t == 'NAME':
            if t.v in EXPR_NAME_MAP: return EXPR_NAME_MAP[t.v]
            if t.v in self._SPACED_EXPR_KW: return ' ' + t.v + ' '
            return t.v
        if t.t == 'OP':
            if t.v == '&&': return ' and '
            if t.v == '||': return ' or '
            return t.v
        return t.v


# ── Public API ────────────────────────────────────────────────────────
def compile_kern(source: str) -> str:
    """Compile Kern source code back to Python."""
    tokens = _lex(source)
    return Parser(tokens).compile()


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    print(compile_kern(src))
