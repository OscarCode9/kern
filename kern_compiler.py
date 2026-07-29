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
  (?P<SELF_FN> fn[ \t]+\. )                                |  # fn .method
  (?P<FSTR>  f(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))  |  # f"..."
  (?P<STR>   (?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'))   |  # "..." '...'
  (?P<NUM>   \d+(?:\.\d+)?(?:[eE][+-]?\d+)?)            |  # numbers
  (?P<OP>    &&|\|\||->>|->|:=|//|\*\*|<<|>>|
             \+=|-=|\*=|/=|//=|%=|\*\*=|\|=|&=|\^=|<<=|>>=|
             ==|!=|<=|>=|[+\-*/%@&|^~<>=!?$])            |  # operators
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
        if k == 'SELF_FN': toks.append(Token('SELF_FN', 'fn'))
        elif k == 'NL': toks.append(Token('NL', '\n'))
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
        self.implicit_self_depth = 0

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
            start = self.pos
            s = self._stmt()
            if self.pos == start:
                raise SyntaxError(
                    f"Unexpected token {self.cur.v!r} at pos {self.pos}"
                )
            if s: lines.append(s)
            self.skip_nl()
        return '\n'.join(lines)

    # ── Block: {stmts}[}] → ":\n    stmt\n    stmt" ────────────────
    def _block(self) -> str:
        self.eat('{')
        self.ind += 1
        ind = self._i()
        stmts = []
        self.skip_nl()
        while self.cur.v != '}' and self.cur.t != 'EOF':
            start = self.pos
            s = self._stmt()
            if self.pos == start:
                raise SyntaxError(
                    f"Unexpected token {self.cur.v!r} at pos {self.pos}"
                )
            if s: stmts.append(ind + s)
            # skip ; and NL between stmts
            while self.cur.v == ';' or self.cur.t == 'NL':
                self.pos += 1
        # Kern v0.4 lets EOF close every still-open statement block. Explicit
        # v0.2/v0.3 braces remain accepted for backwards compatibility.
        if self.cur.v == '}':
            self.eat('}')
        self.ind -= 1
        if not stmts:
            return ':\n' + self._i() + '    pass'
        return ':\n' + '\n'.join(stmts)

    def _suite(self) -> str:
        """Compile either a braced suite or v0.4's one-statement ``:suite``."""
        if self.cur.v == '{':
            return self._block()
        self.eat(':')
        self.ind += 1
        try:
            statement = self._stmt()
            if not statement:
                statement = 'pass'
            return ':\n' + self._i() + statement
        finally:
            self.ind -= 1

    def _skip_continuation_separator(self, words: set[str]) -> None:
        """Consume ``;`` only when it introduces a compound continuation."""
        if self.cur.v == ';' and self.peek().v in words:
            self.eat(';')

    # ── Statements ─────────────────────────────────────────────────
    def _stmt(self) -> str:
        c = self.cur
        v = c.v

        if v == '$':
            return self._out(starred=True)
        if v == ':' and self.peek().v == ':':
            return self._out()
        if c.t == 'SELF_FN':
            return self._fn(False)
        if self._looks_like_bare_fn():
            return self._fn(False, bare=True)

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
        if c.t == 'OP' and v == '>':
            return self._return('>')

        return self._expr_stmt()

    def _looks_like_bare_fn(self, start: int | None = None) -> bool:
        """Whether tokens at ``start`` have the reversible v0.4 def shape."""
        pos = self.pos if start is None else start
        token = self.toks[pos]

        if token.v == '.':
            if (
                self.toks[min(pos + 1, len(self.toks) - 1)].t != 'NAME'
                or self.toks[min(pos + 2, len(self.toks) - 1)].v != '('
            ):
                return False
            open_pos = pos + 2
        else:
            if token.t != 'NAME':
                return False
            # These are real Python statement keywords, so ``if(x){...}``
            # must remain an if statement rather than looking like a def.
            if token.v in {
                'if', 'for', 'while', 'try', 'raise', 'with', 'del',
                'assert', 'pass', 'break', 'continue', 'global',
                'nonlocal', 'async', 'yield', 'from', 'else', 'elif',
            }:
                return False
            if self.toks[min(pos + 1, len(self.toks) - 1)].v != '(':
                return False
            open_pos = pos + 1

        depth = 1
        index = open_pos + 1
        while index < len(self.toks):
            value = self.toks[index].v
            if value in ('(', '[', '{'):
                depth += 1
            elif value in (')', ']', '}'):
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth:
            return False

        index += 1
        if self.toks[min(index, len(self.toks) - 1)].v == '->':
            index += 1
            annotation_depth = 0
            while index < len(self.toks):
                value = self.toks[index].v
                if annotation_depth == 0 and value in ('{', '='):
                    return True
                if value in ('(', '[', '{'):
                    annotation_depth += 1
                elif value in (')', ']', '}'):
                    if annotation_depth == 0:
                        return False
                    annotation_depth -= 1
                if self.toks[index].t in ('NL', 'EOF') and annotation_depth == 0:
                    return False
                index += 1
            return False
        return self.toks[min(index, len(self.toks) - 1)].v in ('{', '=')

    def _fn(self, is_async: bool, bare: bool = False) -> str:
        if bare:
            implicit_self = self.cur.v == '.'
            if implicit_self:
                self.eat('.')
            name = self.eat().v
        else:
            implicit_self = self.cur.t == 'SELF_FN'
            if implicit_self:
                self.eat()
            else:
                self.eat('fn')
            name = self.eat().v
        self.eat('(')
        params = self._params()
        self.eat(')')
        if implicit_self:
            params = 'self' + (', ' + params if params else '')
        ret_ann = ''
        if self.cur.v == '->':
            self.eat('->')
            ret_ann = ' -> ' + self._expr_until({'{', '='})
        prefix = 'async def' if is_async else 'def'
        if implicit_self:
            self.implicit_self_depth += 1
        try:
            if self.cur.v == '=':
                self.eat('=')
                body = self._expr_line()
                return f'{prefix} {name}({params}){ret_ann}:\n{self._i()}    return {body}'
            block = self._block()
        finally:
            if implicit_self:
                self.implicit_self_depth -= 1
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

    def _return(self, marker: str = 'ret') -> str:
        self.eat(marker)
        if self.cur.v in (';', '}') or self.cur.t in ('NL', 'EOF'):
            return 'return'
        # Reversible fusion: >x=expr expands to x=expr; return x.
        if self.cur.t == 'NAME' and self.peek().v == '=':
            name = self.eat().v
            self.eat('=')
            value = self._expr_line()
            return f'{name} = {value}\n{self._i()}return {name}'
        return 'return ' + self._expr_line()

    def _out(self, *, starred: bool = False) -> str:
        """Compile compact output syntax: ``::value`` or ``$iterable``."""
        if starred:
            self.eat('$')
        else:
            self.eat(':')
            self.eat(':')
        value = self._expr_line()
        if not value:
            marker = '$' if starred else '::'
            raise SyntaxError(f"Expected an expression after {marker!r}")
        return f"print({'*' if starred else ''}{value})"

    def _if(self) -> str:
        self.eat('if')
        cond = self._expr_until({'{', ':', ';'})
        block = self._suite()
        s = 'if ' + cond + block
        self._skip_continuation_separator({'elif', 'else'})
        while self.cur.v == 'elif':
            self.eat('elif')
            cond = self._expr_until({'{', ':', ';'})
            s += '\n' + self._i() + 'elif ' + cond + self._suite()
            self._skip_continuation_separator({'elif', 'else'})
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._suite()
        return s

    def _for(self) -> str:
        self.eat('for')
        target = self._expr_until({'in'})
        self.eat('in')
        iter_ = self._expr_until({'{', ':', ';'})
        block = self._suite()
        s = f'for {target} in {iter_}{block}'
        self._skip_continuation_separator({'else'})
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._suite()
        return s

    def _while(self) -> str:
        self.eat('while')
        cond = self._expr_until({'{', ':', ';'})
        block = self._suite()
        s = 'while ' + cond + block
        self._skip_continuation_separator({'else'})
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._suite()
        return s

    def _try(self) -> str:
        self.eat('try')
        s = 'try' + self._suite()
        self._skip_continuation_separator({'exc', 'else', 'fin'})
        while self.cur.v == 'exc':
            self.eat('exc')
            exc_clause = ''
            # exc Type as name or exc(T1,T2) as name
            if self.cur.v == '(':
                self.eat('(')
                types = self._expr_list(')')
                self.eat(')')
                exc_clause = '(' + types + ')'
            elif self.cur.v not in ('{', ':', ';') and self.cur.t != 'NL':
                exc_type = self._expr_until({'as', '{', ':', ';'})
                exc_clause = ' ' + exc_type
            as_name = ''
            if self.cur.v == 'as':
                self.eat('as')
                as_name = ' as ' + self.eat().v
            s += '\n' + self._i() + 'except' + exc_clause + as_name + self._suite()
            self._skip_continuation_separator({'exc', 'else', 'fin'})
        if self.cur.v == 'else':
            self.eat('else')
            s += '\n' + self._i() + 'else' + self._suite()
            self._skip_continuation_separator({'fin'})
        if self.cur.v == 'fin':
            self.eat('fin')
            s += '\n' + self._i() + 'finally' + self._suite()
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
        return 'with ' + items + self._suite()

    def _with_items(self) -> str:
        parts = []
        while True:
            ctx = self._expr_until({'as', ',', '{', ':'})
            if self.cur.v == 'as':
                self.eat('as')
                var = self._expr_until({',', '{', ':'})
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
        if self.cur.v == 'fn' or self.cur.t == 'SELF_FN':
            return self._fn(True)
        if self._looks_like_bare_fn():
            return self._fn(True, bare=True)
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
        delimiters = []
        expect_operand = True
        previous_value = None
        while self.cur.t != 'EOF':
            v = self.cur.v
            # Always check stops before depth tracking
            if not delimiters and v in stops: break
            if not delimiters and self.cur.t == 'NL': break

            # In a direct call argument position, ``:x`` is Kern v0.4's
            # compact spelling of the Python keyword argument ``x=x``.
            # Requiring the innermost delimiter to be ``(`` keeps slices such
            # as ``items[:x]`` untouched.
            if (
                v == ':'
                and self.peek().t == 'NAME'
                and delimiters
                and delimiters[-1] == '('
                and previous_value in ('(', ',')
            ):
                self.eat(':')
                name = self.eat().v
                parts.append(name + '=' + name)
                expect_operand = False
                previous_value = name
                continue

            if v in ('{', '(', '['):
                delimiters.append(v)
            elif v in ('}', ')', ']'):
                if not delimiters:
                    break   # unmatched close — stop
                expected = {')': '(', ']': '[', '}': '{'}[v]
                if delimiters[-1] != expected:
                    raise SyntaxError(
                        f"Mismatched delimiter {v!r} at pos {self.pos}"
                    )
                delimiters.pop()
            # Lambda: \params:body — can appear at any depth
            if v == '\\':
                self.pos += 1
                parts.append(self._lambda_with_stops(stops))
                expect_operand = False
                previous_value = v
                continue
            if v == '~' and not expect_operand:
                self.pos += 1
                parts.append('[::-1]')
                expect_operand = False
                previous_value = v
                continue
            if v == '.' and self.implicit_self_depth and expect_operand:
                self.pos += 1
                if (
                    self.cur.t == 'NAME'
                    and self.cur.v not in self._SPACED_EXPR_KW
                ):
                    parts.append('self.')
                    expect_operand = True
                else:
                    parts.append('self')
                    expect_operand = False
                previous_value = v
                continue
            token = self.cur
            parts.append(self._next_tok())
            previous_value = token.v
            if token.t in ('NAME', 'NUM', 'STR') or token.v in (')', ']', '}'):
                expect_operand = (
                    token.t == 'NAME' and token.v in self._SPACED_EXPR_KW
                )
            elif token.v in ('?', '!'):
                expect_operand = False
            elif token.v == '.':
                expect_operand = True
            elif token.v in ('(', '[', '{', ',', ':'):
                expect_operand = True
            else:
                expect_operand = True
        if self.cur.t == 'EOF' and delimiters:
            raise SyntaxError(
                f"Unclosed delimiter {delimiters[-1]!r} at EOF"
            )
        return ''.join(parts)

    def _lambda_with_stops(self, stops: set) -> str:
        r"""Parse \params:body inheriting the parent expression's stops."""
        # Collect parameter tokens until ':' at depth 0. Defaults are emitted
        # with Python expression keywords, whose required spaces the lexer
        # discards, so rebuild them through the normal token translator.
        param_toks = []
        d = 0
        while self.cur.t != 'EOF':
            v = self.cur.v
            if v == ':' and d == 0: break
            if v in ('(', '[', '{'): d += 1
            elif v in (')', ']', '}'): d -= 1
            param_toks.append(self._next_tok())
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
        'yield', 'lambda',
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
            if t.v == '?': return ' is None'
            if t.v == '!': return ' is not None'
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
