"""Conservative semantic compaction for Kern's optional compact mode.

The normal Kern emitter preserves Python identifiers exactly. Compact mode
keeps public/module names and function parameters stable, but alpha-renames
private locals inside function-like scopes with tokenizer-friendly aliases and
applies guarded return/control-flow simplifications. This gives Kern a
comparison contract similar to a source minifier while leaving the default
reversible mode unchanged.
"""

from __future__ import annotations

import ast
import copy
import keyword
from collections import Counter
from dataclasses import dataclass, field


_INTROSPECTION_NAMES = {"dir", "eval", "exec", "locals", "vars"}
# Measured across cl100k_base and o200k_base on the modern 1,682-program
# corpus.  The order is tokenizer-friendly, but aliases are still rejected
# whenever they collide with names visible in the current or a child scope.
_ALIAS_BASE = tuple("_kxyzijabcmnpqrstuvwlodefghABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class Scope:
    node: ast.AST
    parent: Scope | None
    kind: str
    bindings: set[str] = field(default_factory=set)
    parameters: set[str] = field(default_factory=set)
    globals: set[str] = field(default_factory=set)
    nonlocals: set[str] = field(default_factory=set)
    pattern_bindings: set[str] = field(default_factory=set)
    used_names: set[str] = field(default_factory=set)
    loaded_names: set[str] = field(default_factory=set)
    occurrences: Counter[str] = field(default_factory=Counter)
    mapping: dict[str, str] = field(default_factory=dict)
    unsafe_introspection: bool = False


def _bound_names(target: ast.AST | None) -> set[str]:
    if target is None:
        return set()
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_bound_names(item))
        return names
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return set()


class _ScopeBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scopes: dict[int, Scope] = {}
        self.current: Scope | None = None

    def build(self, tree: ast.Module) -> tuple[ast.Module, dict[int, Scope]]:
        module = Scope(tree, None, "module")
        self.scopes[id(tree)] = module
        self.current = module
        for stmt in tree.body:
            self.visit(stmt)
        self._finish(module)
        return tree, self.scopes

    def _push(self, node: ast.AST, kind: str) -> Scope:
        assert self.current is not None
        scope = Scope(node, self.current, kind)
        self.scopes[id(node)] = scope
        previous = self.current
        self.current = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            self._record_arguments(node.args)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in node.body:
                self.visit(stmt)
        elif isinstance(node, ast.Lambda):
            self.visit(node.body)
        self._finish(scope)
        self.current = previous
        return scope

    def _finish(self, scope: Scope) -> None:
        scope.bindings.difference_update(scope.globals)
        scope.bindings.difference_update(scope.nonlocals)
        if scope.kind not in {"function", "lambda", "comprehension"}:
            return

        # A child scope may use both one of its own renamed bindings and a
        # binding captured from this scope. Keep ancestor aliases distinct
        # from every descendant alias so those two names cannot collapse.
        descendant_aliases: set[str] = set()
        descendant_loaded_names: set[str] = set()
        descendant_uses_introspection = False
        for child in self.scopes.values():
            ancestor = child.parent
            while ancestor is not None and ancestor is not scope:
                ancestor = ancestor.parent
            if ancestor is scope:
                descendant_aliases.update(child.mapping.values())
                descendant_loaded_names.update(child.loaded_names)
                descendant_uses_introspection |= child.unsafe_introspection

        if scope.unsafe_introspection or descendant_uses_introspection:
            return

        unavailable = (
            set(scope.used_names)
            | set(scope.bindings)
            | descendant_aliases
            | descendant_loaded_names
        )
        short_index = 0
        suffixed_index = 0

        def next_alias(original_name: str) -> str:
            nonlocal short_index, suffixed_index
            while short_index < len(_ALIAS_BASE):
                candidate = _ALIAS_BASE[short_index]
                short_index += 1
                if (
                    candidate == "_"
                    and original_name in scope.pattern_bindings
                ):
                    continue
                if candidate not in unavailable:
                    return candidate
            while True:
                candidate = f"A{suffixed_index}"
                suffixed_index += 1
                if candidate not in unavailable:
                    return candidate

        candidates = [
            name
            for name in scope.bindings
            if name not in scope.parameters
            and name not in {"_", "self", "cls"}
            and not (name.startswith("__") and name.endswith("__"))
            and len(name) > 1
            and scope.occurrences[name] > 1
            and not keyword.iskeyword(name)
        ]
        candidates.sort(
            key=lambda name: (
                scope.occurrences[name] * max(1, len(name) - 1),
                len(name),
                name,
            ),
            reverse=True,
        )
        for name in candidates:
            alias = next_alias(name)
            scope.mapping[name] = alias
            unavailable.add(alias)

    def _record_arguments(self, args: ast.arguments) -> None:
        assert self.current is not None
        all_args = (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        )
        for arg in all_args:
            self.current.bindings.add(arg.arg)
            self.current.parameters.add(arg.arg)
            self.current.used_names.add(arg.arg)
            self.current.occurrences[arg.arg] += 1

    def _visit_outer_expression(self, node: ast.AST | None) -> None:
        if node is not None:
            self.visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        assert self.current is not None
        self.current.used_names.add(node.id)
        self.current.occurrences[node.id] += 1
        if isinstance(node.ctx, ast.Load):
            self.current.loaded_names.add(node.id)
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.current.bindings.add(node.id)
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in _INTROSPECTION_NAMES
        ):
            self.current.unsafe_introspection = True

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        assert self.current is not None
        self.current.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        assert self.current is not None
        self.current.nonlocals.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        assert self.current is not None
        self.current.bindings.add(node.name)
        self.current.used_names.add(node.name)
        self.current.occurrences[node.name] += 1
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in list(node.args.defaults) + [
            item for item in node.args.kw_defaults if item is not None
        ]:
            self.visit(default)
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            self._visit_outer_expression(arg.annotation)
        if node.args.vararg:
            self._visit_outer_expression(node.args.vararg.annotation)
        if node.args.kwarg:
            self._visit_outer_expression(node.args.kwarg.annotation)
        self._visit_outer_expression(node.returns)
        self._push(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        for default in list(node.args.defaults) + [
            item for item in node.args.kw_defaults if item is not None
        ]:
            self.visit(default)
        self._push(node, "lambda")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        assert self.current is not None
        self.current.bindings.add(node.name)
        self.current.used_names.add(node.name)
        self.current.occurrences[node.name] += 1
        for item in node.decorator_list + node.bases:
            self.visit(item)
        for item in node.keywords:
            self.visit(item.value)
        scope = Scope(node, self.current, "class")
        self.scopes[id(node)] = scope
        previous = self.current
        self.current = scope
        for stmt in node.body:
            self.visit(stmt)
        self._finish(scope)
        self.current = previous

    def _visit_comprehension_scope(self, node: ast.AST) -> None:
        assert self.current is not None
        generators = node.generators  # type: ignore[attr-defined]
        # Python evaluates the outermost iterable in the containing scope.
        # The target, filters, remaining iterables, and result expression live
        # in the comprehension's implicit function scope.
        self.visit(generators[0].iter)
        scope = Scope(node, self.current, "comprehension")
        self.scopes[id(node)] = scope
        previous = self.current
        self.current = scope
        for index, generator in enumerate(generators):
            scope.bindings.update(_bound_names(generator.target))
            self.visit(generator.target)
            if index:
                self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)  # type: ignore[attr-defined]
        self._finish(scope)
        self.current = previous

    visit_ListComp = _visit_comprehension_scope
    visit_SetComp = _visit_comprehension_scope
    visit_DictComp = _visit_comprehension_scope
    visit_GeneratorExp = _visit_comprehension_scope

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        assert self.current is not None
        for item in node.names:
            name = item.asname or item.name.split(".", 1)[0]
            self.current.bindings.add(name)
            self.current.used_names.add(name)
            self.current.occurrences[name] += 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        assert self.current is not None
        for item in node.names:
            name = item.asname or item.name
            self.current.bindings.add(name)
            self.current.used_names.add(name)
            self.current.occurrences[name] += 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        assert self.current is not None
        if node.name:
            self.current.bindings.add(node.name)
            self.current.used_names.add(node.name)
            self.current.occurrences[node.name] += 1
        self._visit_outer_expression(node.type)
        for stmt in node.body:
            self.visit(stmt)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        assert self.current is not None
        if self.current.kind == "comprehension":
            # Assignment expressions in comprehensions bind in the nearest
            # containing non-comprehension scope. Disabling both mappings is
            # conservative and preserves that special symbol-table behavior.
            scope: Scope | None = self.current
            while scope is not None:
                scope.unsafe_introspection = True
                if scope.kind != "comprehension":
                    break
                scope = scope.parent
        self.visit(node.value)
        self.visit(node.target)

    def _record_pattern_name(self, name: str | None) -> None:
        if not name:
            return
        assert self.current is not None
        self.current.bindings.add(name)
        self.current.pattern_bindings.add(name)
        self.current.used_names.add(name)
        self.current.occurrences[name] += 1

    def visit_MatchAs(self, node: ast.MatchAs) -> None:  # noqa: N802
        if node.pattern is not None:
            self.visit(node.pattern)
        self._record_pattern_name(node.name)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:  # noqa: N802
        self._record_pattern_name(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:  # noqa: N802
        for key in node.keys:
            self.visit(key)
        for pattern in node.patterns:
            self.visit(pattern)
        self._record_pattern_name(node.rest)


class _LocalRenamer(ast.NodeTransformer):
    def __init__(self, scopes: dict[int, Scope]) -> None:
        self.scopes = scopes
        self.stack: list[Scope] = []

    @property
    def current(self) -> Scope:
        return self.stack[-1]

    def _resolve(self, name: str) -> str:
        for scope in reversed(self.stack):
            if scope.kind == "class" and scope is not self.current:
                continue
            if name in scope.globals:
                return name
            if name in scope.bindings:
                return scope.mapping.get(name, name)
        return name

    def rename(self, tree: ast.Module) -> ast.Module:
        self.stack.append(self.scopes[id(tree)])
        tree = self.visit(tree)
        self.stack.pop()
        ast.fix_missing_locations(tree)
        return tree

    def visit_Name(self, node: ast.Name) -> ast.Name:  # noqa: N802
        node.id = self._resolve(node.id)
        return node

    def visit_Global(self, node: ast.Global) -> ast.Global:  # noqa: N802
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Nonlocal:  # noqa: N802
        node.names = [self._resolve(name) for name in node.names]
        return node

    def _visit_arguments_outer(self, args: ast.arguments) -> None:
        args.defaults = [self.visit(item) for item in args.defaults]
        args.kw_defaults = [
            self.visit(item) if item is not None else None
            for item in args.kw_defaults
        ]
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        ):
            if arg.annotation is not None:
                arg.annotation = self.visit(arg.annotation)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        node.name = self.current.mapping.get(node.name, node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        self._visit_arguments_outer(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        self.stack.append(self.scopes[id(node)])
        node.body = [self.visit(stmt) for stmt in node.body]
        self.stack.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:  # noqa: N802
        self._visit_arguments_outer(node.args)
        self.stack.append(self.scopes[id(node)])
        node.body = self.visit(node.body)
        self.stack.pop()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # noqa: N802
        node.name = self.current.mapping.get(node.name, node.name)
        node.decorator_list = [self.visit(item) for item in node.decorator_list]
        node.bases = [self.visit(item) for item in node.bases]
        node.keywords = [
            ast.keyword(arg=item.arg, value=self.visit(item.value))
            for item in node.keywords
        ]
        self.stack.append(self.scopes[id(node)])
        node.body = [self.visit(stmt) for stmt in node.body]
        self.stack.pop()
        return node

    def _visit_comprehension(self, node: ast.AST) -> ast.AST:
        generators = node.generators  # type: ignore[attr-defined]
        generators[0].iter = self.visit(generators[0].iter)
        self.stack.append(self.scopes[id(node)])
        return self._finish_comprehension(node)

    def _finish_comprehension(self, node: ast.AST) -> ast.AST:
        generators = node.generators  # type: ignore[attr-defined]
        for index, generator in enumerate(generators):
            generator.target = self.visit(generator.target)
            if index:
                generator.iter = self.visit(generator.iter)
            generator.ifs = [self.visit(item) for item in generator.ifs]
        if isinstance(node, ast.DictComp):
            node.key = self.visit(node.key)
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)  # type: ignore[attr-defined]
        self.stack.pop()
        return node

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_Import(self, node: ast.Import) -> ast.Import:  # noqa: N802
        for item in node.names:
            bound = item.asname or item.name.split(".", 1)[0]
            renamed = self.current.mapping.get(bound)
            if renamed:
                item.asname = renamed
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:  # noqa: N802
        for item in node.names:
            bound = item.asname or item.name
            renamed = self.current.mapping.get(bound)
            if renamed:
                item.asname = renamed
        return node

    def visit_ExceptHandler(
        self, node: ast.ExceptHandler
    ) -> ast.ExceptHandler:  # noqa: N802
        if node.type is not None:
            node.type = self.visit(node.type)
        if node.name:
            node.name = self.current.mapping.get(node.name, node.name)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.MatchAs:  # noqa: N802
        if node.pattern is not None:
            node.pattern = self.visit(node.pattern)
        if node.name:
            node.name = self.current.mapping.get(node.name, node.name)
        return node

    def visit_MatchStar(
        self, node: ast.MatchStar
    ) -> ast.MatchStar:  # noqa: N802
        if node.name:
            node.name = self.current.mapping.get(node.name, node.name)
        return node

    def visit_MatchMapping(
        self, node: ast.MatchMapping
    ) -> ast.MatchMapping:  # noqa: N802
        node.keys = [self.visit(key) for key in node.keys]
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        if node.rest:
            node.rest = self.current.mapping.get(node.rest, node.rest)
        return node


class _CompactSimplifier(ast.NodeTransformer):
    """Apply behavior-preserving simplifications for compact mode only."""

    def __init__(self) -> None:
        self._functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self._try_descendants: set[int] = set()
        self._range_shadowed = False

    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        self._range_shadowed = (
            "range" in self._string_bindings(node)
            or any(
                (
                    isinstance(item, ast.Name)
                    and isinstance(item.ctx, (ast.Store, ast.Del))
                    and item.id == "range"
                )
                or (isinstance(item, ast.arg) and item.arg == "range")
                for item in ast.walk(node)
            )
        )
        return self.generic_visit(node)

    @staticmethod
    def _uses_introspection(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        return any(
            isinstance(item, ast.Name)
            and isinstance(item.ctx, ast.Load)
            and item.id in _INTROSPECTION_NAMES
            for item in ast.walk(node)
        )

    @staticmethod
    def _string_bindings(node: ast.AST) -> set[str]:
        names: set[str] = set()
        for item in ast.walk(node):
            if isinstance(item, (ast.MatchAs, ast.MatchStar)) and item.name:
                names.add(item.name)
            elif isinstance(item, ast.MatchMapping) and item.rest:
                names.add(item.rest)
            elif isinstance(item, ast.ExceptHandler) and item.name:
                names.add(item.name)
            elif isinstance(
                item,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                names.add(item.name)
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                names.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in item.names
                )
        return names

    def _mark_try_descendants(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for item in ast.walk(node):
            if isinstance(item, (ast.Try, ast.TryStar)):
                self._try_descendants.update(
                    id(descendant) for descendant in ast.walk(item)
                )

    def _can_inline_assign_return(
        self,
        assignment: ast.stmt,
        returned: ast.stmt,
    ) -> bool:
        if not self._functions:
            return False
        if id(assignment) in self._try_descendants:
            return False
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and isinstance(returned, ast.Return)
            and isinstance(returned.value, ast.Name)
            and returned.value.id == assignment.targets[0].id
        ):
            return False

        function = self._functions[-1]
        target = assignment.targets[0].id
        parameters = {
            arg.arg
            for arg in (
                list(function.args.posonlyargs)
                + list(function.args.args)
                + list(function.args.kwonlyargs)
                + ([function.args.vararg] if function.args.vararg else [])
                + ([function.args.kwarg] if function.args.kwarg else [])
            )
        }
        if target in parameters or target in self._string_bindings(function):
            return False

        occurrences = [
            item
            for item in ast.walk(function)
            if isinstance(item, ast.Name) and item.id == target
        ]
        return (
            len(occurrences) == 2
            and occurrences[0] in {
                assignment.targets[0],
                returned.value,
            }
            and occurrences[1] in {
                assignment.targets[0],
                returned.value,
            }
            and not any(
                isinstance(item, (ast.Global, ast.Nonlocal))
                and target in item.names
                for item in ast.walk(function)
            )
        )

    def _inline_assign_returns(
        self,
        statements: list[ast.stmt],
    ) -> list[ast.stmt]:
        rewritten: list[ast.stmt] = []
        index = 0
        while index < len(statements):
            if (
                index + 1 < len(statements)
                and self._can_inline_assign_return(
                    statements[index],
                    statements[index + 1],
                )
            ):
                assignment = statements[index]
                returned = statements[index + 1]
                assert isinstance(assignment, ast.Assign)
                assert isinstance(returned, ast.Return)
                rewritten.append(
                    ast.copy_location(
                        ast.Return(value=assignment.value),
                        returned,
                    )
                )
                index += 2
                continue
            rewritten.append(statements[index])
            index += 1
        return rewritten

    def generic_visit(self, node: ast.AST) -> ast.AST:
        visited = super().generic_visit(node)
        if self._functions:
            for field, value in ast.iter_fields(visited):
                if (
                    isinstance(value, list)
                    and value
                    and all(isinstance(item, ast.stmt) for item in value)
                ):
                    setattr(
                        visited,
                        field,
                        self._inline_assign_returns(value),
                    )
        return visited

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        if self._uses_introspection(node):
            return node
        self._mark_try_descendants(node)
        self._functions.append(node)
        visited = self.generic_visit(node)
        self._functions.pop()
        return visited

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    @staticmethod
    def _literal_int(node: ast.AST) -> int | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            return -node.operand.value
        return None

    def _range_mod_filter(self, node: ast.GeneratorExp) -> ast.Call | None:
        """Replace a guarded range-modulo filter with an exact stepped range."""
        if self._range_shadowed or len(node.generators) != 1:
            return None
        generator = node.generators[0]
        if generator.is_async or len(generator.ifs) != 1:
            return None
        if not (
            isinstance(generator.target, ast.Name)
            and isinstance(node.elt, ast.Name)
            and node.elt.id == generator.target.id
            and isinstance(generator.iter, ast.Call)
            and isinstance(generator.iter.func, ast.Name)
            and generator.iter.func.id == "range"
            and not generator.iter.keywords
            and len(generator.iter.args) in {1, 2}
        ):
            return None

        if len(generator.iter.args) == 1:
            start_node = ast.Constant(value=0)
            stop_node = generator.iter.args[0]
        else:
            start_node, stop_node = generator.iter.args
        start = _CompactSimplifier._literal_int(start_node)
        stop = _CompactSimplifier._literal_int(stop_node)
        if start is None or stop is None:
            return None

        condition = generator.ifs[0]
        if not (
            isinstance(condition, ast.Compare)
            and len(condition.ops) == 1
            and isinstance(condition.ops[0], ast.Eq)
            and len(condition.comparators) == 1
        ):
            return None
        candidates = (condition.left, condition.comparators[0])
        modulo = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, ast.BinOp)
                and isinstance(candidate.op, ast.Mod)
                and isinstance(candidate.left, ast.Name)
                and candidate.left.id == generator.target.id
                and isinstance(candidate.right, ast.Constant)
                and isinstance(candidate.right.value, int)
                and not isinstance(candidate.right.value, bool)
            ),
            None,
        )
        zero = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, ast.Constant)
                and candidate.value == 0
                and not isinstance(candidate.value, bool)
            ),
            None,
        )
        if modulo is None or zero is None:
            return None
        divisor = modulo.right.value
        if divisor <= 0:
            return None

        first = start + (-start) % divisor
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Constant(value=first),
                    copy.deepcopy(stop_node),
                    ast.Constant(value=divisor),
                ],
                keywords=[],
            ),
            node,
        )

    def visit_GeneratorExp(
        self,
        node: ast.GeneratorExp,
    ) -> ast.GeneratorExp | ast.Call:
        node = self.generic_visit(node)
        assert isinstance(node, ast.GeneratorExp)
        return self._range_mod_filter(node) or node

    def visit_Return(self, node: ast.Return) -> ast.Return:  # noqa: N802
        node = self.generic_visit(node)
        if (
            self._functions
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        ):
            node.value = None
        return node

    def visit_If(self, node: ast.If) -> ast.If | list[ast.stmt]:  # noqa: N802
        node = self.generic_visit(node)
        if (
            self._functions
            and node.orelse
            and node.body
            and isinstance(node.body[-1], (ast.Return, ast.Raise))
        ):
            continuation = node.orelse
            node.orelse = []
            return [node, *continuation]
        return node


def compact_locals(tree: ast.Module) -> ast.Module:
    """Return a copied AST with conservative function-local alpha-renaming."""
    copied = copy.deepcopy(tree)
    copied, scopes = _ScopeBuilder().build(copied)
    return _LocalRenamer(scopes).rename(copied)


def compact_tree(tree: ast.Module) -> ast.Module:
    """Return the exact semantic AST used by Kern compact mode."""
    compacted = compact_locals(tree)
    compacted = _CompactSimplifier().visit(compacted)
    combined: list[ast.stmt] = []
    index = 0
    while index < len(compacted.body):
        node = compacted.body[index]
        if isinstance(node, ast.Import):
            names = list(node.names)
            index += 1
            while (
                index < len(compacted.body)
                and isinstance(compacted.body[index], ast.Import)
            ):
                names.extend(compacted.body[index].names)
                index += 1
            combined.append(ast.copy_location(ast.Import(names=names), node))
            continue
        combined.append(node)
        index += 1
    compacted.body = combined
    ast.fix_missing_locations(compacted)
    return compacted
