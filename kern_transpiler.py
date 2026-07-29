"""
kern_transpiler.py — Python → Kern
Convierte código Python a representación Kern compacta usando el módulo ast.

Grammar spec v0.4:
  name(params)=expr             single-expression function
  name(params){stmts}           multi-statement function
  .method(params){.attr}        implicit self receiver
  call(:x)                      same-name keyword argument (x=x)
  if cond:stmt                  one-simple-statement suite
  EOF                           closes any still-open statement blocks
  if cond{stmts}elif...else{}   conditionals
  for x in iter{stmts}          for loops
  while cond{stmts}             while loops
  imp module / from mod imp x   imports
  cls Name(Base){stmts}         classes
  try{...}exc Type{...}fin{}    try/except/finally
  \\params:expr                 lambda
  >expr                         return
  >x=expr                       assign x, then return x
  x? / x!                       x is None / x is not None
  x=expr, x+=expr               assignments
  x>0&&y<0  x||y                and→&& or→||
"""

import ast
import sys


# Internal-only marker for a statement-block close. Real expression braces keep
# using ``}``, so only structural closes at the very end can be omitted safely.
_BLOCK_CLOSE = "\x00"


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
    ast.IfExp: 0,       # x if c else y
    ast.BoolOp: 1,      # overridden for and / or
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

    def __init__(self):
        self._implicit_self_depth = 0
        self._fstring_depth = 0
        self._compact_mode = False

    def transpile(self, source: str, compact: bool = False) -> str:
        self._compact_mode = compact
        tree = ast.parse(source)
        if compact:
            from kern_compact import compact_tree

            tree = compact_tree(tree)
        nodes = self._strip_nonsemantic_string_exprs(list(tree.body))
        parts = []
        for node in nodes:
            parts.append(self._stmt(node))
        rendered = "\n".join(p for p in parts if p)
        # EOF closes the final chain of statement blocks. Any earlier block
        # marker remains explicit so the following statement is unambiguous.
        rendered = rendered.rstrip(_BLOCK_CLOSE)
        return rendered.replace(_BLOCK_CLOSE, "}")

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
        """Render statements with compact, structurally unambiguous separators."""
        nodes = [s for s in stmts if not self._is_nop_string_expr_stmt(s)]
        rendered = []
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if i + 1 < len(nodes) and self._can_fuse_assign_return(node, nodes[i + 1]):
                target = node.targets[0].id
                rendered.append(
                    (">" + target + "=" + self._expr_tuple_bare(node.value), node)
                )
                i += 2
                continue
            text = self._stmt(node)
            if text:
                rendered.append((text, node))
            i += 1

        out = []
        for i, (text, node) in enumerate(rendered):
            out.append(text)
            if i + 1 < len(rendered) and not self._is_self_delimiting_stmt(node, text):
                out.append(";")
        return "".join(out)

    def _can_fuse_assign_return(self, node, next_node) -> bool:
        """Whether ``x=expr; return x`` can use the reversible ``>x=expr`` form."""
        return (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(next_node, ast.Return)
            and isinstance(next_node.value, ast.Name)
            and next_node.value.id == node.targets[0].id
        )

    def _is_self_delimiting_stmt(self, node, text: str) -> bool:
        """Statement blocks delimit themselves without a following semicolon."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = self._strip_pass(
                self._strip_nonsemantic_string_exprs(list(node.body))
            )
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and body[0].value is not None
            ):
                return False
        return (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.With,
                    ast.AsyncWith,
                ),
            )
            # During emission every real statement-block close is represented
            # by the private marker. A literal ``}`` here can instead belong
            # to a dict/set expression in an inline suite.
            and text.endswith(_BLOCK_CLOSE)
        )

    def _strip_leading_docstring(self, stmts):
        """Drop only the first statement if it is a docstring."""
        if (stmts
                and isinstance(stmts[0], ast.Expr)
                and isinstance(stmts[0].value, ast.Constant)
                and isinstance(stmts[0].value.value, str)):
            return stmts[1:]
        return stmts

    def _is_nop_string_expr_stmt(self, node) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    def _strip_nonsemantic_string_exprs(self, stmts):
        """Drop bare string expression statements (docstrings / no-op literals)."""
        return [s for s in stmts if not self._is_nop_string_expr_stmt(s)]

    def _strip_pass(self, stmts):
        """Drop Pass statements — empty {} body handled by compiler."""
        return [s for s in stmts if not isinstance(s, ast.Pass)]

    def _expr_tuple_bare(self, node) -> str:
        """Emit multi-element tuples without outer parens (for targets/returns)."""
        if isinstance(node, ast.Tuple) and len(node.elts) > 1:
            return ",".join(self._expr(e) for e in node.elts)
        return self._expr(node)

    def _block(self, stmts) -> str:
        """Render {stmts} block."""
        inner = self._stmts(stmts)
        return "{" + inner + _BLOCK_CLOSE

    def _suite(self, stmts, has_continuation: bool = False) -> str:
        """Render a safe one-statement suite with ``:``; otherwise use braces."""
        nodes = [s for s in stmts if not self._is_nop_string_expr_stmt(s)]
        compound = (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.If,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.Try,
            ast.With,
            ast.AsyncWith,
        )
        if len(nodes) == 1 and not isinstance(nodes[0], compound):
            text = self._stmt(nodes[0])
            if text:
                return ":" + text + (";" if has_continuation else "")
        return self._block(stmts)

    def _stmt_FunctionDef(self, node) -> str:
        return self._fn(node, is_async=False)

    def _stmt_AsyncFunctionDef(self, node) -> str:
        return self._fn(node, is_async=True)

    def _fn(self, node, is_async: bool) -> str:
        prefix = "async " if is_async else ""
        implicit_self = self._can_use_implicit_self(node.args)
        name = ("." if implicit_self else "") + node.name
        params = self._args(node.args, skip_first=implicit_self)
        ret_ann = ""
        if node.returns:
            ret_ann = "->" + self._header_expr(node.returns)

        decorators = "".join("@" + self._expr(d) + "\n" for d in node.decorator_list)

        # Real body: drop bare string literal expressions and lone pass stmts.
        body = self._strip_pass(self._strip_nonsemantic_string_exprs(list(node.body)))

        if implicit_self:
            self._implicit_self_depth += 1
        try:
            if not body:
                body_str = self._block([])
            elif len(body) == 1 and isinstance(body[0], ast.Return):
                # Single-expression form: fn f(x)=expr
                val = body[0].value
                if val is not None:
                    body_str = "=" + self._expr(val)
                else:
                    body_str = self._block(body)
            else:
                body_str = self._block(body)
        finally:
            if implicit_self:
                self._implicit_self_depth -= 1

        return f"{decorators}{prefix}{name}({params}){ret_ann}{body_str}"

    def _can_use_implicit_self(self, args) -> bool:
        if args.posonlyargs or not args.args or args.args[0].arg != "self":
            return False
        first = args.args[0]
        first_has_default = len(args.defaults) == len(args.args)
        return first.annotation is None and not first_has_default

    def _args(self, args, skip_first: bool = False) -> str:
        parts = []
        # positional args with defaults aligned from the right
        positional = list(args.posonlyargs) + list(args.args)
        n_defaults = len(args.defaults)
        n_args = len(positional)
        for i, arg in enumerate(positional):
            default_idx = i - (n_args - n_defaults)
            s = arg.arg
            if arg.annotation:
                s += ":" + self._expr(arg.annotation)
            if default_idx >= 0:
                s += "=" + self._expr(args.defaults[default_idx])
            if skip_first and i == 0:
                continue
            parts.append(s)
            if i + 1 == len(args.posonlyargs):
                parts.append("/")
        # *args
        if args.vararg:
            s = "*" + args.vararg.arg
            if args.vararg.annotation:
                s += ":" + self._expr(args.vararg.annotation)
            parts.append(s)
        # Preserve Python keyword-only separator when there is no *args.
        if args.kwonlyargs and not args.vararg:
            parts.append("*")
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
            return ">"
        return ">" + self._expr_tuple_bare(node.value)

    def _stmt_Assign(self, node) -> str:
        parts = [self._expr_tuple_bare(t) for t in node.targets]
        parts.append(self._expr_tuple_bare(node.value))
        return "=".join(parts)

    def _stmt_AnnAssign(self, node) -> str:
        s = self._expr(node.target) + ":" + self._expr(node.annotation)
        if node.value:
            s += "=" + self._expr(node.value)
        return s

    def _stmt_AugAssign(self, node) -> str:
        op = AUGOP[type(node.op)]
        return self._expr(node.target) + op + self._expr(node.value)

    def _stmt_If(self, node) -> str:
        parts = [
            "if "
            + self._header_expr(node.test)
            + self._suite(node.body, has_continuation=bool(node.orelse))
        ]
        # Flatten elif chains
        orelse = node.orelse
        while orelse:
            if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                inner = orelse[0]
                parts.append(
                    "elif "
                    + self._header_expr(inner.test)
                    + self._suite(
                        inner.body,
                        has_continuation=bool(inner.orelse),
                    )
                )
                orelse = inner.orelse
            else:
                parts.append("else" + self._suite(orelse))
                break
        return "".join(parts)

    def _stmt_For(self, node) -> str:
        target = self._expr_tuple_bare(node.target)
        iter_ = self._header_expr(node.iter)
        body = self._suite(node.body, has_continuation=bool(node.orelse))
        s = f"for {target} in {iter_}{body}"
        if node.orelse:
            s += "else" + self._suite(node.orelse)
        return s

    def _stmt_AsyncFor(self, node) -> str:
        target = self._expr_tuple_bare(node.target)
        iter_ = self._header_expr(node.iter)
        body = self._suite(node.body, has_continuation=bool(node.orelse))
        s = f"async for {target} in {iter_}{body}"
        if node.orelse:
            s += "else" + self._suite(node.orelse)
        return s

    def _stmt_While(self, node) -> str:
        body = self._suite(node.body, has_continuation=bool(node.orelse))
        s = "while " + self._header_expr(node.test) + body
        if node.orelse:
            s += "else" + self._suite(node.orelse)
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
        # Drop bare string literal expressions and lone pass in class body.
        body = self._strip_pass(self._strip_nonsemantic_string_exprs(list(node.body)))
        body_str = self._block(body)
        return f"{decorators}cls {node.name}{base_str}{body_str}"

    def _stmt_Try(self, node) -> str:
        continuations = len(node.handlers) + bool(node.orelse) + bool(node.finalbody)
        s = "try" + self._suite(
            node.body,
            has_continuation=bool(continuations),
        )
        for index, handler in enumerate(node.handlers):
            exc_str = "exc"
            if handler.type:
                if isinstance(handler.type, ast.Tuple):
                    types = ",".join(self._expr(t) for t in handler.type.elts)
                    exc_str += "(" + types + ")"
                else:
                    exc_str += " " + self._header_expr(handler.type)
                if handler.name:
                    exc_str += " as " + handler.name
            remaining = (
                len(node.handlers) - index - 1
                + bool(node.orelse)
                + bool(node.finalbody)
            )
            s += exc_str + self._suite(
                handler.body,
                has_continuation=bool(remaining),
            )
        if node.orelse:
            s += "else" + self._suite(
                node.orelse,
                has_continuation=bool(node.finalbody),
            )
        if node.finalbody:
            s += "fin" + self._suite(node.finalbody)
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
        return f"with {items}" + self._suite(node.body)

    def _stmt_AsyncWith(self, node) -> str:
        items = ",".join(self._withitem(i) for i in node.items)
        return f"async with {items}" + self._suite(node.body)

    def _withitem(self, item) -> str:
        s = self._header_expr(item.context_expr)
        if item.optional_vars:
            s += " as " + self._expr(item.optional_vars)
        return s

    def _header_expr(self, node) -> str:
        """Protect a leading expression brace from block/suite lookahead."""
        rendered = self._expr(node)
        if rendered.startswith("{"):
            return "(" + rendered + ")"
        return rendered

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
        # Bare expression statement. Ignore no-op string literal statements.
        if self._is_nop_string_expr_stmt(node):
            return ""
        if (
            self._compact_mode
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
            and len(node.value.args) == 1
            and not node.value.keywords
        ):
            argument = node.value.args[0]
            if isinstance(argument, ast.Starred):
                return "$" + self._expr(argument.value)
            return "::" + self._expr(argument)
        return self._expr(node.value)

    # yield / yield from as statements
    def _stmt_Yield(self, node) -> str:
        if node.value:
            return "yield " + self._expr(node.value)
        return "yield"

    # ── Expressions ────────────────────────────────────────────────

    def _expr(self, node) -> str:
        method = "_expr_" + node.__class__.__name__
        handler = getattr(self, method, None)
        if handler:
            return handler(node)
        return f"<{node.__class__.__name__}>"

    def _expr_Constant(self, node) -> str:
        if node.value is Ellipsis:
            return "..."
        if (
            isinstance(node.value, float)
            and 0 <= node.value < 1
        ):
            rendered = repr(node.value)
            if rendered.startswith("0."):
                return rendered[1:]
        return repr(node.value)

    def _expr_Name(self, node) -> str:
        if (
            node.id == "self"
            and self._implicit_self_depth
            and not self._fstring_depth
        ):
            return "."
        return node.id

    def _expr_Attribute(self, node) -> str:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and self._implicit_self_depth
            and not self._fstring_depth
        ):
            return "." + node.attr
        return self._expr(node.value) + "." + node.attr

    def _expr_Subscript(self, node) -> str:
        if (
            self._compact_mode
            and isinstance(node.slice, ast.Slice)
            and node.slice.lower is None
            and node.slice.upper is None
            and isinstance(node.slice.step, ast.UnaryOp)
            and isinstance(node.slice.step.op, ast.USub)
            and isinstance(node.slice.step.operand, ast.Constant)
            and node.slice.step.operand.value == 1
        ):
            return self._expr_with_parens(node.value, node) + "~"
        # In subscripts, tuple slices must be emitted without parentheses:
        # arr[:,0] instead of arr[(:,0)].
        if isinstance(node.slice, ast.Tuple):
            if not node.slice.elts:
                inner = "()"
            elif len(node.slice.elts) == 1:
                inner = "(" + self._expr(node.slice.elts[0]) + ",)"
            else:
                inner = ",".join(self._expr(e) for e in node.slice.elts)
        else:
            inner = self._expr(node.slice)
        return self._expr(node.value) + "[" + inner + "]"

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
        if isinstance(node, ast.BoolOp):
            return 2 if isinstance(node.op, ast.And) else 1
        if isinstance(node, ast.BinOp):
            return BINOP_PREC.get(type(node.op), 6)
        return PREC.get(type(node), 15)

    def _expr_UnaryOp(self, node) -> str:
        op_str = UNARYOP.get(type(node.op), "?")
        operand = self._expr_with_parens(node.operand, node)
        return op_str + operand

    def _expr_BoolOp(self, node) -> str:
        if self._fstring_depth:
            op_str = " and " if isinstance(node.op, ast.And) else " or "
        else:
            op_str = BOOLOP[type(node.op)]
        parts = []
        for value in node.values:
            # Preserve explicit grouping even for a nested identical BoolOp.
            if isinstance(value, ast.BoolOp):
                part = "(" + self._expr(value) + ")"
            else:
                part = self._expr_with_parens(value, node)
            parts.append(part)
        return op_str.join(parts)

    def _expr_Compare(self, node) -> str:
        s = self._compare_operand(node.left)
        for op, comp in zip(node.ops, node.comparators):
            if (
                not self._fstring_depth
                and isinstance(comp, ast.Constant)
                and comp.value is None
            ):
                if isinstance(op, ast.Is):
                    s += "?"
                    continue
                if isinstance(op, ast.IsNot):
                    s += "!"
                    continue
            s += CMPOP.get(type(op), "?") + self._compare_operand(comp)
        return s

    def _compare_operand(self, node) -> str:
        s = self._expr(node)
        if self._node_prec(node) < PREC[ast.Compare] or isinstance(node, ast.Compare):
            return "(" + s + ")"
        return s

    def _expr_Call(self, node) -> str:
        func = self._expr(node.func)
        if isinstance(node.func, ast.Lambda):
            func = "(" + func + ")"
        all_args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                all_args.append("*" + self._expr(a.value))
                continue
            if isinstance(a, ast.Lambda):
                all_args.append("(" + self._expr(a) + ")")
                continue
            # list((x for ...)) -> list(x for ...) when generator is sole arg.
            if (
                isinstance(a, ast.GeneratorExp)
                and len(node.args) == 1
                and len(node.keywords) == 0
            ):
                all_args.append(self._expr_generator_inner(a))
                continue
            all_args.append(self._expr(a))
        for keyword in node.keywords:
            if keyword.arg is None:
                all_args.append("**" + self._expr(keyword.value))
            elif (
                not self._fstring_depth
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == keyword.arg
            ):
                all_args.append(":" + keyword.arg)
            else:
                value = self._expr(keyword.value)
                if isinstance(keyword.value, ast.Lambda):
                    value = "(" + value + ")"
                all_args.append(keyword.arg + "=" + value)
        return func + "(" + ",".join(all_args) + ")"

    def _expr_Starred(self, node) -> str:
        return "*" + self._expr(node.value)

    def _expr_IfExp(self, node) -> str:
        # Keep Python ternary: value if test else orelse
        return (self._expr(node.body) + " if " +
                self._expr(node.test) + " else " +
                self._expr(node.orelse))

    def _expr_Lambda(self, node) -> str:
        # Lambda parameters are parsed as a compact raw segment by the inverse
        # compiler, so keep every expression in defaults Python-compatible.
        implicit_self_depth = self._implicit_self_depth
        fstring_depth = self._fstring_depth
        self._implicit_self_depth = 0
        self._fstring_depth += 1
        try:
            params = self._args(node.args)
        finally:
            self._implicit_self_depth = implicit_self_depth
            self._fstring_depth = fstring_depth
        body = self._expr(node.body)
        if self._fstring_depth:
            return "(lambda " + params + ":" + body + ")"
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
        return "(" + self._expr_generator_inner(node) + ")"

    def _expr_generator_inner(self, node) -> str:
        return self._expr(node.elt) + self._comprehensions(node.generators)

    def _comprehensions(self, generators) -> str:
        s = ""
        for gen in generators:
            s += " for " + self._expr_tuple_bare(gen.target) + " in " + self._expr(gen.iter)
            for cond in gen.ifs:
                s += " if " + self._expr(cond)
        return s

    def _expr_JoinedStr(self, node) -> str:
        # f-string: reconstruct as f"..."
        parts = []
        self._fstring_depth += 1
        try:
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
        finally:
            self._fstring_depth -= 1
        return 'f"' + "".join(parts) + '"'

    def _expr_Await(self, node) -> str:
        return "await " + self._expr(node.value)

    def _expr_Yield(self, node) -> str:
        if node.value:
            return "yield " + self._expr(node.value)
        return "yield"

    def _expr_YieldFrom(self, node) -> str:
        return "yield from " + self._expr(node.value)

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

def transpile(source: str, compact: bool = False) -> str:
    """Convert Python to Kern, optionally alpha-renaming private locals."""
    return KernEmitter().transpile(source, compact=compact)


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    print(transpile(src))
