from __future__ import annotations

import ast
import unittest

from kern_compact import compact_tree
from kern_compiler import compile_kern
from kern_transpiler import transpile


def compile_namespace(source: str, *, compact: bool) -> tuple[str, dict]:
    kern = transpile(source, compact=compact)
    python_back = compile_kern(kern)
    namespace: dict = {}
    exec(python_back, namespace)
    return kern, namespace


def compact_namespace(source: str) -> dict:
    namespace: dict = {}
    exec(ast.unparse(compact_tree(ast.parse(source))), namespace)
    return namespace


class CompactModeTests(unittest.TestCase):
    def test_default_mode_keeps_original_identifiers(self) -> None:
        source = """
def calculate(value):
    descriptive_total = value + 1
    return descriptive_total * descriptive_total
"""
        kern = transpile(source)
        self.assertIn("descriptive_total", kern)
        self.assertEqual(
            ast.dump(ast.parse(source), include_attributes=False),
            ast.dump(ast.parse(compile_kern(kern)), include_attributes=False),
        )

    def test_compact_mode_keeps_public_api_and_renames_locals(self) -> None:
        source = """
def calculate(value, scale=2):
    descriptive_total = value + 1
    repeated_result = descriptive_total * descriptive_total
    return repeated_result * scale
"""
        kern, namespace = compile_namespace(source, compact=True)
        self.assertTrue(kern.startswith("calculate(value,scale=2)"))
        self.assertNotIn("descriptive_total", kern)
        self.assertNotIn("repeated_result", kern)
        self.assertEqual(namespace["calculate"](3, scale=3), 48)

    def test_nested_closure_resolution(self) -> None:
        source = """
def outer(value):
    long_multiplier = value + 2
    def nested(number):
        return number * long_multiplier
    return nested(4)
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["outer"](3), 20)

    def test_comprehension_scope_can_reuse_short_names(self) -> None:
        source = """
def transform(values):
    transformed_values = [current_value * 2 for current_value in values]
    return transformed_values
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["transform"]([1, 2, 3]), [2, 4, 6])

    def test_comprehension_outer_iterable_uses_parent_scope(self) -> None:
        source = """
def transform(values):
    current_value = [2, 3, 4]
    transformed_values = [current_value * 2 for current_value in current_value]
    return transformed_values
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["transform"](None), [4, 6, 8])

    def test_comprehension_alias_does_not_shadow_captured_local(self) -> None:
        source = """
def unique(values):
    occurrence_counts = {}
    for current_value in values:
        occurrence_counts[current_value] = occurrence_counts.get(current_value, 0) + 1
    return [
        current_value
        for current_value in values
        if occurrence_counts[current_value] == 1
    ]
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["unique"]([1, 2, 2, 3]), [1, 3])

    def test_comprehension_walrus_keeps_containing_scope_binding(self) -> None:
        source = """
def capture():
    latest_value = -1
    collected_values = [
        (latest_value := current_value)
        for current_value in range(3)
    ]
    return latest_value, collected_values
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["capture"](), (2, [0, 1, 2]))

    def test_method_closure_skips_class_namespace(self) -> None:
        source = """
def build():
    captured_value = 7
    class Container:
        captured_value = 11
        class_copy = captured_value
        def read(self):
            return captured_value
    return Container.class_copy, Container().read()
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["build"](), (11, 7))

    def test_match_pattern_bindings_are_renamed_consistently(self) -> None:
        source = """
def unpack(value):
    descriptive_value = None
    match value:
        case {"item": descriptive_value, **remaining_values}:
            return descriptive_value, remaining_values
    return descriptive_value, {}
"""
        namespace = compact_namespace(source)
        self.assertEqual(
            namespace["unpack"]({"item": 4, "other": 9}),
            (4, {"other": 9}),
        )

    def test_more_than_twenty_six_locals_get_unique_aliases(self) -> None:
        assignments = "\n".join(
            f"    descriptive_value_{index} = {index}"
            for index in range(30)
        )
        expression = " + ".join(
            f"descriptive_value_{index}" for index in range(30)
        )
        source = f"def total():\n{assignments}\n    return {expression}\n"
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["total"](), sum(range(30)))

    def test_local_import_alias_is_rewritten_consistently(self) -> None:
        source = """
def root(value):
    import math
    calculated_result = math.sqrt(value)
    return calculated_result
"""
        _, namespace = compile_namespace(source, compact=True)
        self.assertEqual(namespace["root"](81), 9)

    def test_adjacent_top_level_imports_match_compact_reference_ast(self) -> None:
        source = "import math\nimport statistics\nanswer = math.sqrt(81)\n"
        decoded = compile_kern(transpile(source, compact=True))
        self.assertEqual(
            ast.dump(compact_tree(ast.parse(source)), include_attributes=False),
            ast.dump(ast.parse(decoded), include_attributes=False),
        )

    def test_locals_introspection_disables_scope_renaming(self) -> None:
        source = """
def inspect_local():
    descriptive_value = 7
    captured_names = locals()
    return descriptive_value, "descriptive_value" in captured_names
"""
        kern, namespace = compile_namespace(source, compact=True)
        self.assertIn("descriptive_value", kern)
        self.assertEqual(namespace["inspect_local"](), (7, True))

    def test_float_spelling_compacts_without_value_change(self) -> None:
        source = """
def ratio():
    return 0.5
"""
        kern, namespace = compile_namespace(source, compact=True)
        self.assertIn("=.5", kern)
        self.assertEqual(namespace["ratio"](), 0.5)


if __name__ == "__main__":
    unittest.main()
