"""
kern_transpiler.py — Python → Kern
Convierte código Python a representación Kern compacta usando el módulo ast.

Grammar spec v0.2:
  fn name(params)=expr          single-expression function
  fn name(params){stmts}        multi-statement function
  if cond{stmts}elif...else{}   conditionals
  for x in iter{stmts}          for loops
  while cond{stmts}             while loops
  imp module / from mod imp x   imports
  cls Name(Base){stmts}         classes
  try{...}exc Type{...}fin{}    try/except/finally
  \params:expr                  lambda
  ret expr                      return
  x=expr, x+=expr               assignments
  x>0&y<0  x|y                  and→& or→|
"""

import ast
import sys


# ── Operator maps ──────────────────────────────────────────────────
BINOP = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
    ast.LShift: "<<", ast.RShift: ">>",
    # Bitwise keep original symbols — unambiguous since BoolOp uses && / ||
    ast.BitOr: "|", ast.BitAnd: "&", ast.BitXor: "^",
    ast.MatMult: "@",
}

# Logical ops: && / || — distinct from bitwise & / |
BOOLOP = {
    ast.And: "&&",
    ast.Or: "||",
}

CMPOP = {
    ast.Eq: "==", ast.NotEq: "!=",
    ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=",
    ast.Is: " is ", ast.IsNot: " is not ",
    ast.In: " in ", ast.NotIn: " not in ",
}

UNARYOP = {
    ast.USub: "-", ast.UAdd: "+",
    ast.Invert: "~", ast.Not: "not ",
}

AUGOP = {
    ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=",
    ast.FloorDiv: "//=", ast.Mod: "%=", ast.Pow: "**=",
    ast.BitOr: "|=", ast.BitAnd: "&=", ast.BitXor: "^=",
    ast.LShift: "<<=", ast.RShift: ">>=",
}

# Precedence for non-BinOp nodes
PREC = {
    ast.BoolOp: 1,      # and / or
    ast.IfExp: 2,       # x if c else y
    ast.Compare: 5,     # == != < > etc.
    ast.BinOp: 6,       # base (overridden per-op below)
    ast.UnaryOp: 13,
    ast.Await: 13,
    ast.Call: 15,
    ast.Attribute: 15,
    ast.Subscript: 15,
}

# Per-operator precedence for BinOp nodes
BINOP_PREC = {
    ast.BitOr:    6,
    ast.BitXor:   7,
    ast.BitAnd:   8,
    ast.LShift:   9,  ast.RShift:   9,
    ast.Add:      10, ast.Sub:      10,
    ast.Mult:     11, ast.Div:      11,
    ast.FloorDiv: 11, ast.Mod:      11, ast.MatMult: 11,
    ast.Pow:      14,  # right-associative
}


class KernEmitter(ast.NodeVisitor):

    def transpile(self, source: str) -> str:
        tree = ast.parse(source)
        parts = []
        for node in self._strip_leading_docstring(list(tree.body)):
            parts.append(self._stmt(node))
        return "\n".join(p for p in parts if p)

    # ── Statements ─────────────────────────────────────────────────

    def _stmt(self, node) -> str:
        method = "_stmt_" + node.__class__.__name__
        handler = getattr(self, method, None)
        if handler:
            return handler(node)
        # Fallback: expression statement (e.g. bare function call)
        if isinstance(node, ast.Expr):
            return self._expr(node.value)
        return f"# UNSUPPORTED:{node.__class__.__name__}"

    def _stmts(self, stmts) -> str:
        """Render a list of statements, skipping docstrings."""
        parts = []
        for i, s in enumerate(stmts):
            # Skip leading docstring
            if i == 0 and isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue
            parts.append(self._stmt(s))
        return ";".join(p for p in parts if p)

    def _strip_leading_docstring(self, stmts):
        """Drop only the first statement if it is a docstring."""
        if (stmts
                and isinstance(stmts[0], ast.Expr)
                and isinstance(stmts[0].value, ast.Constant)
                and isinstance(stmts[0].value.value, str)):
            return stmts[1:]
        return stmts

    def _block(self, stmts) -> str:
        """Render {stmts} block."""
        inner = self._stmts(stmts)
        return "{" + inner + "}"

    def _stmt_FunctionDef(self, node) -> str:
        return self._fn(node, is_async=False)

    def _stmt_AsyncFunctionDef(self, node) -> str:
        return self._fn(node, is_async=True)

    def _fn(self, node, is_async: bool) -> str:
        prefix = "async fn" if is_async else "fn"
        name = node.name
        params = self._args(node.args)
        ret_ann = ""
        if node.returns:
            ret_ann = "->" + self._expr(node.returns)

        decorators = "".join("@" + self._expr(d) + "\n" for d in node.decorator_list)

        # Real body: skip only leading docstring
        body = self._strip_leading_docstring(list(node.body))

        if not body:
            body_str = "{}"
        elif len(body) == 1 and isinstance(body[0], ast.Return):
            # Single-expression form: fn f(x)=expr
            val = body[0].value
            if val is not None:
                body_str = "=" + self._expr(val)
            else:
                body_str = "{}"
        else:
            body_str = self._block(body)

        return f"{decorators}{prefix} {name}({params}){ret_ann}{body_str}"

    def _args(self, args) -> str:
        parts = []
        # positional args with defaults aligned from the right
        n_defaults = len(args.defaults)
        n_args = len(args.args)
        for i, arg in enumerate(args.args):
            default_idx = i - (n_args - n_defaults)
            s = arg.arg
            if arg.annotation:
                s += ":" + self._expr(arg.annotation)
            if default_idx >= 0:
                s += "=" + self._expr(args.defaults[default_idx])
            parts.append(s)
        # *args
        if args.vararg:
            s = "*" + args.vararg.arg
            if args.vararg.annotation:
                s += ":" + self._expr(args.vararg.annotation)
            parts.append(s)
        # keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            s = arg.arg
            if arg.annotation:
                s += ":" + self._expr(arg.annotation)
            if args.kw_defaults[i] is not None:
                s += "=" + self._expr(args.kw_defaults[i])
            parts.append(s)
        # **kwargs
        if args.kwarg:
            s = "**" + args.kwarg.arg
            if args.kwarg.annotation:
                s += ":" + self._expr(args.kwarg.annotation)
            parts.append(s)
        return ",".join(parts)

    def _stmt_Return(self, node) -> str:
        if node.value is None:
            return "ret"
        return "ret " + self._expr(node.value)

    def _stmt_Assign(self, node) -> str:
        targets = ",".join(self._expr(t) for t in node.targets)
        return targets + "=" + self._expr(node.value)

    def _stmt_AnnAssign(self, node) -> str:
        s = self._expr(node.target) + ":" + self._expr(node.annotation)
        if node.value:
            s += "=" + self._expr(node.value)
        return s

    def _stmt_AugAssign(self, node) -> str:
        op = AUGOP[type(node.op)]
        return self._expr(node.target) + op + self._expr(node.value)

    def _stmt_If(self, node) -> str:
        parts = ["if " + self._expr(node.test) + self._block(node.body)]
        # Flatten elif chains
        orelse = node.orelse
        while orelse:
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                inner = orelse[0]
                parts.append("elif " + self._expr(inner.test) + self._block(inner.body))
                orelse = inner.orelse
            else:
                parts.append("else" + self._block(orelse))
                break
        return "".join(parts)

    def _stmt_For(self, node) -> str:
        target = self._expr(node.target)
        iter_ = self._expr(node.iter)
        body = self._block(node.body)
        s = f"for {target} in {iter_}{body}"
        if node.orelse:
            s += "else" + self._block(node.orelse)
        return s

    def _stmt_While(self, node) -> str:
        body = self._block(node.body)
        s = "while " + self._expr(node.test) + body
        if node.orelse:
            s += "else" + self._block(node.orelse)
        return s

    def _stmt_Import(self, node) -> str:
        parts = []
        for alias in node.names:
            s = alias.name
            if alias.asname:
                s += " as " + alias.asname
            parts.append(s)
        return "imp " + ",".join(parts)

    def _stmt_ImportFrom(self, node) -> str:
        mod = "." * (node.level or 0) + (node.module or "")
        names = ",".join(
            (a.name + " as " + a.asname) if a.asname else a.name
            for a in node.names
        )
        return f"from {mod} imp {names}"

    def _stmt_ClassDef(self, node) -> str:
        bases = ",".join(self._expr(b) for b in node.bases)
        base_str = f"({bases})" if bases else ""
        decorators = "".join("@" + self._expr(d) + "\n" for d in node.decorator_list)
        # Skip only leading docstring in class body
        body = self._strip_leading_docstring(list(node.body))
        body_str = self._block(body) if body else "{}"
        return f"{decorators}cls {node.name}{base_str}{body_str}"

    def _stmt_Try(self, node) -> str:
        s = "try" + self._block(node.body)
        for handler in node.handlers:
            exc_str = "exc"
            if handler.type:
                if isinstance(handler.type, ast.Tuple):
                    types = ",".join(self._expr(t) for t in handler.type.elts)
                    exc_str += "(" + types + ")"
                else:
                    exc_str += " " + self._expr(handler.type)
                if handler.name:
                    exc_str += " as " + handler.name
            s += exc_str + self._block(handler.body)
        if node.orelse:
            s += "else" + self._block(node.orelse)
        if node.finalbody:
            s += "fin" + self._block(node.finalbody)
        return s

    def _stmt_Raise(self, node) -> str:
        s = "raise"
        if node.exc:
            s += " " + self._expr(node.exc)
        if node.cause:
            s += " from " + self._expr(node.cause)
        return s

    def _stmt_With(self, node) -> str:
        items = ",".join(self._withitem(i) for i in node.items)
        return f"with {items}" + self._block(node.body)

    def _withitem(self, item) -> str:
        s = self._expr(item.context_expr)
        if item.optional_vars:
            s += " as " + self._expr(item.optional_vars)
        return s

    def _stmt_Delete(self, node) -> str:
        return "del " + ",".join(self._expr(t) for t in node.targets)

    def _stmt_Assert(self, node) -> str:
        s = "assert " + self._expr(node.test)
        if node.msg:
            s += "," + self._expr(node.msg)
        return s

    def _stmt_Pass(self, node) -> str:
        return "pass"

    def _stmt_Break(self, node) -> str:
        return "break"

    def _stmt_Continue(self, node) -> str:
        return "continue"

    def _stmt_Global(self, node) -> str:
        return "global " + ",".join(node.names)

    def _stmt_Nonlocal(self, node) -> str:
        return "nonlocal " + ",".join(node.names)

    def _stmt_Expr(self, node) -> str:
        # Bare expression statement
        return self._expr(node.value)

    # yield / yield from as statements
    def _stmt_Yield(self, node) -> str:
        if node.value:
            return "yld " + self._expr(node.value)
        return "yld"

    # ── Expressions ────────────────────────────────────────────────

    def _expr(self, node) -> str:
        method = "_expr_" + node.__class__.__name__
        handler = getattr(self, method, None)
        if handler:
            return handler(node)
        return f"<{node.__class__.__name__}>"

    def _expr_Constant(self, node) -> str:
        return repr(node.value)

    def _expr_Name(self, node) -> str:
        return node.id

    def _expr_Attribute(self, node) -> str:
        return self._expr(node.value) + "." + node.attr

    def _expr_Subscript(self, node) -> str:
        return self._expr(node.value) + "[" + self._expr(node.slice) + "]"

    def _expr_Index(self, node) -> str:  # Python 3.8 compat
        return self._expr(node.value)

    def _expr_Slice(self, node) -> str:
        lower = self._expr(node.lower) if node.lower else ""
        upper = self._expr(node.upper) if node.upper else ""
        step  = (":" + self._expr(node.step)) if node.step else ""
        return f"{lower}:{upper}{step}"

    def _expr_BinOp(self, node) -> str:
        op_str = BINOP.get(type(node.op), "?")
        my_prec = BINOP_PREC.get(type(node.op), 6)
        is_pow = isinstance(node.op, ast.Pow)

        # Left child: needs parens if its prec < my prec
        lp = self._node_prec(node.left)
        left_s = self._expr(node.left)
        if lp < my_prec:
            left_s = "(" + left_s + ")"

        # Right child: ** is right-associative, others need parens if prec <= my prec
        rp = self._node_prec(node.right)
        right_s = self._expr(node.right)
        threshold = my_prec if is_pow else my_prec
        if rp < threshold or (not is_pow and rp == my_prec and isinstance(node.right, ast.BinOp)):
            right_s = "(" + right_s + ")"

        return left_s + op_str + right_s

    def _node_prec(self, node) -> int:
        """Return the effective precedence of an expression node."""
        if isinstance(node, ast.BinOp):
            return BINOP_PREC.get(type(node.op), 6)
        return PREC.get(type(node), 15)

    def _expr_UnaryOp(self, node) -> str:
        op_str = UNARYOP.get(type(node.op), "?")
        operand = self._expr_with_parens(node.operand, node)
        return op_str + operand

    def _expr_BoolOp(self, node) -> str:
        op_str = BOOLOP[type(node.op)]
        parts = [self._expr_with_parens(v, node) for v in node.values]
        return op_str.join(parts)

    def _expr_Compare(self, node) -> str:
        s = self._expr(node.left)
        for op, comp in zip(node.ops, node.comparators):
            s += CMPOP.get(type(op), "?") + self._expr(comp)
        return s

    def _expr_Call(self, node) -> str:
        func = self._expr(node.func)
        args = [self._expr(a) for a in node.args]
        kwargs = [k.arg + "=" + self._expr(k.value) if k.arg else "**" + self._expr(k.value)
                  for k in node.keywords]
        stars = ["*" + self._expr(a.value) if isinstance(a, ast.Starred) else self._expr(a)
                 for a in node.args]
        all_args = stars + kwargs
        return func + "(" + ",".join(all_args) + ")"

    def _expr_Starred(self, node) -> str:
        return "*" + self._expr(node.value)

    def _expr_IfExp(self, node) -> str:
        # Keep Python ternary: value if test else orelse
        return (self._expr(node.body) + " if " +
                self._expr(node.test) + " else " +
                self._expr(node.orelse))

    def _expr_Lambda(self, node) -> str:
        params = self._args(node.args)
        body = self._expr(node.body)
        return "\\" + params + ":" + body

    def _expr_List(self, node) -> str:
        return "[" + ",".join(self._expr(e) for e in node.elts) + "]"

    def _expr_Tuple(self, node) -> str:
        if not node.elts:
            return "()"
        inner = ",".join(self._expr(e) for e in node.elts)
        # Single-element tuple needs trailing comma
        if len(node.elts) == 1:
            inner += ","
        return "(" + inner + ")"

    def _expr_Set(self, node) -> str:
        return "{" + ",".join(self._expr(e) for e in node.elts) + "}"

    def _expr_Dict(self, node) -> str:
        pairs = []
        for k, v in zip(node.keys, node.values):
            if k is None:
                pairs.append("**" + self._expr(v))
            else:
                pairs.append(self._expr(k) + ":" + self._expr(v))
        return "{" + ",".join(pairs) + "}"

    def _expr_ListComp(self, node) -> str:
        return "[" + self._expr(node.elt) + self._comprehensions(node.generators) + "]"

    def _expr_SetComp(self, node) -> str:
        return "{" + self._expr(node.elt) + self._comprehensions(node.generators) + "}"

    def _expr_DictComp(self, node) -> str:
        return ("{" + self._expr(node.key) + ":" + self._expr(node.value)
                + self._comprehensions(node.generators) + "}")

    def _expr_GeneratorExp(self, node) -> str:
        return "(" + self._expr(node.elt) + self._comprehensions(node.generators) + ")"

    def _comprehensions(self, generators) -> str:
        s = ""
        for gen in generators:
            s += " for " + self._expr(gen.target) + " in " + self._expr(gen.iter)
            for cond in gen.ifs:
                s += " if " + self._expr(cond)
        return s

    def _expr_JoinedStr(self, node) -> str:
        # f-string: reconstruct as f"..."
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                inner = self._expr(v.value)
                fmt = ""
                if v.format_spec:
                    fmt = ":" + "".join(
                        str(x.value) if isinstance(x, ast.Constant) else self._expr(x)
                        for x in v.format_spec.values
                    )
                conv = ""
                if v.conversion == ord('s'):
                    conv = "!s"
                elif v.conversion == ord('r'):
                    conv = "!r"
                elif v.conversion == ord('a'):
                    conv = "!a"
                parts.append("{" + inner + conv + fmt + "}")
        return 'f"' + "".join(parts) + '"'

    def _expr_Await(self, node) -> str:
        return "await " + self._expr(node.value)

    def _expr_Yield(self, node) -> str:
        if node.value:
            return "yld " + self._expr(node.value)
        return "yld"

    def _expr_YieldFrom(self, node) -> str:
        return "yld from " + self._expr(node.value)

    def _expr_NamedExpr(self, node) -> str:  # walrus :=
        return self._expr(node.target) + ":=" + self._expr(node.value)

    # ── Parens helper ──────────────────────────────────────────────

    def _expr_with_parens(self, child, parent) -> str:
        child_prec = self._node_prec(child)
        parent_prec = self._node_prec(parent)
        s = self._expr(child)
        if child_prec < parent_prec:
            return "(" + s + ")"
        return s


# ── Public API ─────────────────────────────────────────────────────

def transpile(source: str) -> str:
    """Convert Python source code to Kern."""
    return KernEmitter().transpile(source)


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    print(transpile(src))
