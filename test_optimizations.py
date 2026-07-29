"""Regression tests for Kern v0.3/v0.4 token optimizations."""

from __future__ import annotations

import ast
import unittest

from kern_compact import compact_tree
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

    def test_compact_output_reverse_and_stepped_range(self) -> None:
        source = """\
text = "kern"
print(text[::-1])
print(*(x for x in range(1, 21) if x % 2 == 0))
"""
        kern = transpile(source, compact=True)
        rebuilt = compile_kern(kern)
        expected = ast.unparse(compact_tree(ast.parse(source)))

        self.assertIn("::text~", kern)
        self.assertIn("~", kern)
        self.assertIn("$!2:21:2", kern)
        self.assertEqual(
            ast.dump(ast.parse(rebuilt), include_attributes=False),
            ast.dump(ast.parse(expected), include_attributes=False),
        )
        namespace: dict[str, object] = {}
        exec(rebuilt, namespace)
        self.assertEqual(transpile("print('x')"), "print('x')")
        self.assertEqual(compile_kern("out(1)"), "out(1)")

    def test_compact_array_primitives_are_exactly_reversible(self) -> None:
        source = """\
import math
print(sum(range(1, 101)))
print(math.factorial(10))
print(*sorted([3, 1, 2]))
print(*dict.fromkeys([3, 1, 3, 2]))
print(*(x * x for x in range(1, 5)))
print("abracadabra".count("a"))
print(sum(a * b for a, b in zip([1, 2], [3, 4])))
print(math.gcd(2706, 410))
values = [1, 2, 3, 4, 5]
print(*(values[3:] + values[:3]))
"""
        kern = transpile(source, compact=True)
        rebuilt = compile_kern(kern)
        expected = ast.unparse(compact_tree(ast.parse(source)))

        for marker in (
            "+/!1:101",
            "%10",
            "$^[3,1,2]",
            "$?[3,1,3,2]",
            "$*x:!1:5",
            "'abracadabra'#'a'",
            "@1,2:3,4",
            "&2706:410",
            "$values<<<3",
        ):
            with self.subTest(marker=marker):
                self.assertIn(marker, kern)
        self.assertNotIn("imp math", kern)
        self.assertTrue(rebuilt.startswith("import math\n"))
        self.assertEqual(
            ast.dump(ast.parse(rebuilt), include_attributes=False),
            ast.dump(ast.parse(expected), include_attributes=False),
        )

    def test_math_import_elision_requires_an_encoded_math_primitive(self) -> None:
        source = """\
import math
print(math.sin(1))
"""
        kern = transpile(source, compact=True)

        self.assertIn("imp math", kern)
        self.assertEqual(
            ast.dump(ast.parse(compile_kern(kern)), include_attributes=False),
            ast.dump(
                ast.parse(ast.unparse(compact_tree(ast.parse(source)))),
                include_attributes=False,
            ),
        )

    def test_compact_primitives_do_not_capture_near_matches(self) -> None:
        source = """\
print(sum(value for value in data))
print(sorted(data, reverse=True))
print(math.gcd(*values))
print(sum(a + b for a, b in zip(left, right)))
print(*(x * y for x, y in pairs))
print(*(values[3:] + other[:3]))
"""
        kern = transpile(source, compact=True)

        self.assertNotIn("+/", kern)
        self.assertNotIn("$^", kern)
        self.assertNotIn("&", kern)
        self.assertNotIn("@a,b:", kern)
        self.assertNotIn("*x:", kern)
        self.assertNotIn("<<<", kern)

        boolean_dot = """\
print(sum(a * b for a, b in zip([True, False], [1, 2])))
"""
        boolean_dot_kern = transpile(boolean_dot, compact=True)
        self.assertIn(
            "@a,b:[True,False]:[1,2]",
            boolean_dot_kern,
        )
        self.assertEqual(
            ast.dump(
                ast.parse(compile_kern(boolean_dot_kern)),
                include_attributes=False,
            ),
            ast.dump(
                compact_tree(ast.parse(boolean_dot)),
                include_attributes=False,
            ),
        )

    def test_existing_none_and_bitwise_syntax_remains_distinct(self) -> None:
        source = """\
def f(value, other):
    return value is not None and (value ^ other)
"""
        kern = transpile(source, compact=True)
        rebuilt = compile_kern(kern)
        expected = ast.unparse(compact_tree(ast.parse(source)))

        self.assertEqual(
            ast.dump(ast.parse(rebuilt), include_attributes=False),
            ast.dump(ast.parse(expected), include_attributes=False),
        )

    def test_palindrome_and_additive_recurrence_primitives_roundtrip(self) -> None:
        source = """\
text = "racecar"
print(int(text == text[::-1]))
values = [0, 1]
for _ in range(10):
    values.append(values[-1] + values[-2])
print(*values)
"""
        kern = transpile(source, compact=True)
        rebuilt = compile_kern(kern)
        expected = ast.unparse(compact_tree(ast.parse(source)))

        self.assertIn("::=~text", kern)
        self.assertIn("$values=[0,1]\\10", kern)
        self.assertEqual(
            ast.dump(ast.parse(rebuilt), include_attributes=False),
            ast.dump(ast.parse(expected), include_attributes=False),
        )
        namespace: dict[str, object] = {}
        exec(rebuilt, namespace)
        self.assertEqual(namespace["values"][-1], 89)

    def test_recurrence_primitive_requires_the_exact_loop_shape(self) -> None:
        source = """\
values = [0, 1]
for index in range(10):
    values.append(values[-1] + values[-2])
print(*values)
"""
        kern = transpile(source, compact=True)

        self.assertNotIn("$values=[0,1]\\10", kern)

    def test_stepped_range_rewrite_is_guarded(self) -> None:
        exact = ast.unparse(
            compact_tree(
                ast.parse(
                    "values = "
                    "(x for x in range(-3, 5) if 0 == x % 2)"
                )
            )
        )
        dynamic = ast.unparse(
            compact_tree(
                ast.parse(
                    "values = "
                    "(x for x in range(start, stop) if x % 2 == 0)"
                )
            )
        )
        negative_divisor = ast.unparse(
            compact_tree(
                ast.parse(
                    "values = "
                    "(x for x in range(1, 10) if x % -2 == 0)"
                )
            )
        )
        shadowed = ast.unparse(
            compact_tree(
                ast.parse(
                    "range = custom_range\n"
                    "values = "
                    "(x for x in range(1, 10) if x % 2 == 0)"
                )
            )
        )

        self.assertIn("range(-2, 5, 2)", exact)
        self.assertIn("for x in range(start, stop)", dynamic)
        self.assertIn("for x in range(1, 10)", negative_divisor)
        self.assertIn("for x in range(1, 10)", shadowed)

    def test_fn_identifier_attribute_is_not_receiver_syntax(self) -> None:
        rebuilt = compile_kern("fn.foo()")
        self.assertEqual(rebuilt, "fn.foo()")


if __name__ == "__main__":
    unittest.main()
