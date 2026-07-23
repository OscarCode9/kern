"""Regression tests for Kern v0.3/v0.4 token optimizations."""

from __future__ import annotations

import ast
import unittest

from kern_compiler import compile_kern
from kern_transpiler import transpile


def ast_shape(source: str) -> str:
    return ast.dump(ast.parse(source), include_attributes=False)


class KernOptimizationTests(unittest.TestCase):
    def assert_roundtrip(self, source: str) -> tuple[str, str]:
        kern = transpile(source)
        rebuilt = compile_kern(kern)
        self.assertEqual(ast_shape(source), ast_shape(rebuilt))
        return kern, rebuilt

    def test_postfix_none_checks(self) -> None:
        source = """\
def classify(value):
    if value is None:
        return "missing"
    if value is not None:
        return value
    return
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("value?", kern)
        self.assertIn("value!", kern)

    def test_compact_and_fused_returns(self) -> None:
        source = """\
def build():
    result = make_result()
    return result
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertEqual(kern, "build(){>result=make_result()")

    def test_implicit_self_receiver(self) -> None:
        source = """\
def update(self, value):
    self.value = value
    current = self
    return current.value
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertEqual(kern, ".update(value){.value=value;current=.;>current.value")

    def test_implicit_self_in_lambda_default_and_fstring(self) -> None:
        source = """\
def render(self):
    callback = lambda owner=wrap(self): owner.name
    return f"{self.name} {self.name is None} {self.name and 'ok'}"
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn(r"\owner=wrap(self):owner.name", kern)
        self.assertIn(
            'f"{self.name} {self.name is None} {self.name and \'ok\'}"',
            kern,
        )

    def test_compound_statements_are_self_delimiting(self) -> None:
        source = """\
def countdown(value):
    if value:
        value -= 1
    while value:
        value -= 1
    return value
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertEqual(
            kern,
            "countdown(value){if value:value-=1;while value:value-=1;>value",
        )

    def test_boolean_grouping_and_not_precedence(self) -> None:
        source = """\
def check(a, b, c):
    left = a and (b or c)
    right = not a == b
    nested = a or (b or c)
    return left, right, nested
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("a&&(b||c)", kern)
        self.assertIn("not (a==b)", kern)
        self.assertIn("a||(b||c)", kern)

    def test_empty_and_single_tuple_subscripts(self) -> None:
        source = """\
def lookup(data, key):
    empty = data[()]
    single = data[(key,)]
    return empty, single
"""
        self.assert_roundtrip(source)

    def test_single_expression_nested_function_keeps_separator(self) -> None:
        source = """\
def outer():
    def inner(value):
        return {"value": value}
    return inner(1)
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn('inner(value)={\'value\':value};>inner(1)', kern)

    def test_v04_bare_defs_async_defs_and_implicit_eof_closes(self) -> None:
        source = """\
def first(value):
    return value

async def second(self, value):
    if value:
        self.value = value
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertEqual(
            kern,
            "first(value)=value\nasync .second(value){if value:.value=value",
        )
        self.assertNotIn("fn ", kern)

    def test_same_name_keyword_arguments_do_not_conflict_with_slices(self) -> None:
        source = """\
def dispatch(x, z, data):
    part = data[:x]
    return send(x=x, y=x, z=z, part=part)
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("send(:x,y=x,:z,:part)", kern)
        self.assertIn("data[:x]", kern)

    def test_empty_final_block_can_close_at_eof(self) -> None:
        source = """\
def placeholder():
    pass
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertEqual(kern, "placeholder(){")

    def test_inline_suites_with_all_continuation_families(self) -> None:
        source = """\
def flow(flag, values, manager):
    if flag < 0:
        result = -1
    elif flag == 0:
        result = 0
    else:
        result = 1
    for value in values:
        consume(value)
    else:
        finish()
    while flag:
        flag -= 1
    else:
        stopped()
    try:
        risky()
    except ValueError:
        recover()
    except TypeError:
        recover_type()
    else:
        succeed()
    finally:
        cleanup()
    with manager:
        use(manager)
    return result
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn(
            "if flag<0:result=-1;elif flag==0:result=0;else:result=1",
            kern,
        )
        self.assertIn("for value in values:consume(value);else:finish()", kern)
        self.assertIn("try:risky();exc ValueError:recover()", kern)

    def test_nested_if_uses_braces_to_avoid_dangling_else(self) -> None:
        source = """\
def nested(outer, inner):
    if outer:
        if inner:
            return 1
    else:
        return 2
    return 3
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("if outer{if inner:>1}else:>2", kern)

    def test_inline_suite_expression_colons_remain_contextual(self) -> None:
        source = """\
def contextual(flag, x, data):
    if flag:
        result = send(x=x, part=data[:x], mapping={"x": x})
    return result
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn('if flag:result=send(:x,part=data[:x],mapping={\'x\':x})', kern)

    def test_lambda_boundaries_defaults_and_fstring_stay_python_safe(self) -> None:
        source = """\
def transform(xs, reverse, x):
    callback = lambda y=make(x=x, missing=x is None): y
    ordered = sorted(xs, key=lambda item: item[0], reverse=reverse)
    rendered = f"{make(x=x)}"
    return callback, ordered, rendered
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn(r"key=(\item:item[0]),:reverse", kern)
        self.assertIn("make(x=x,missing=x is None)", kern)
        self.assertIn('f"{make(x=x)}"', kern)

    def test_positional_only_return_none_and_called_lambda(self) -> None:
        source = """\
def choose(x: int, /, y: str = "a") -> str:
    return y

def stop():
    return

def apply(value):
    return (lambda item: item + 1)(value)
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("choose(x:int,/,y:str='a')->str=y", kern)
        self.assertIn("stop(){>", kern)
        self.assertIn(r"apply(value)=(\item:item+1)(value)", kern)

    def test_brace_headers_and_async_compounds(self) -> None:
        source = """\
def brace_headers(value) -> {"field": int}:
    if {1, 2}:
        return {"field": value}
    for item in {"field": value}:
        return {"field": item}

async def consume(stream, manager):
    async for item in stream:
        use(item)
    async with manager:
        use(manager)
"""
        kern, _ = self.assert_roundtrip(source)
        self.assertIn("->({'field':int})", kern)
        self.assertIn("if ({1,2}):", kern)
        self.assertIn("async for item in stream:use(item)", kern)

    def test_invalid_or_unbalanced_closers_fail_fast(self) -> None:
        for kern in ("}", ")", "]", "f(){>1}}", "f(){x=(1"):
            with self.subTest(kern=kern):
                with self.assertRaises(SyntaxError):
                    compile_kern(kern)

    def test_v02_syntax_remains_accepted(self) -> None:
        old_kern = "fn f(self,x){if x is None{ret self.x};ret x}"
        rebuilt = compile_kern(old_kern)
        ast.parse(rebuilt)
        namespace: dict[str, object] = {}
        exec(rebuilt, namespace)
        holder = type("Holder", (), {"x": 7})()
        self.assertEqual(namespace["f"](holder, None), 7)
        self.assertEqual(namespace["f"](holder, 3), 3)

    def test_v03_explicit_closes_remain_accepted(self) -> None:
        old_kern = "fn outer(){fn inner(x)=x;ret inner(1)}"
        rebuilt = compile_kern(old_kern)
        namespace: dict[str, object] = {}
        exec(rebuilt, namespace)
        self.assertEqual(namespace["outer"](), 1)

    def test_fn_identifier_attribute_is_not_receiver_syntax(self) -> None:
        rebuilt = compile_kern("fn.foo()")
        self.assertEqual(rebuilt, "fn.foo()")


if __name__ == "__main__":
    unittest.main()
