# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REPL semantics for one generated code cell.

``run_cell_source`` implements the ``wrap_in_function=True`` contract that both
execution backends share: an async wrapper that captures the last-expression or
explicit ``return`` value, persists locals into the namespace, registers the cell
with ``linecache`` for accurate tracebacks, and surfaces ``return_result()`` as
an ``ExecutionSignal``.

It works on a plain namespace dict and depends only on ``nooa.events`` and the
standard library. It is deliberately not part of ``runtime.sandbox``: the sandbox
is one adapter of this contract, not its owner. The other adapter is
``ActorRuntime.execute_code`` in ``nooa.runtime.actor``, which adds the
in-process concerns this module leaves out — contextvar-backed stream capture for
concurrent cells, async-safety patching, and ``asyncio.wait_for`` timeouts.

The two adapters must produce identical observable results; ``tests/runtime/
test_cell_contract.py`` runs one corpus through both and fails on any divergence.
Keeping the wrapper string aligned between them preserves ``wrapper_line_offset``
so error line numbers match regardless of backend.
"""

from __future__ import annotations

import ast
import contextlib
import io
import linecache
import logging
import tokenize
import types
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal

logger = logging.getLogger(__name__)


class _HookedBuffer(io.StringIO):
    """StringIO that also forwards each write to an optional streaming hook."""

    def __init__(self, hook: Callable[[str], None] | None = None):
        super().__init__()
        self._hook = hook

    def write(self, s: str) -> int:  # type: ignore[override]
        if self._hook and s:
            try:
                self._hook(s)
            except Exception:
                pass
        return super().write(s)


def _has_top_level_return(tree: ast.Module) -> bool:
    """True if the module body has a ``return`` outside any nested def/class."""

    class _V(ast.NodeVisitor):
        found = False

        def visit_Return(self, node: ast.Return) -> None:
            self.found = True

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            pass

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            pass

    v = _V()
    v.visit(tree)
    return v.found


def _indent(code: str, prefix: str) -> str:
    """Indent every line of ``code``, except the insides of multiline strings.

    The cell is indented so it can sit inside the async wrapper. A multiline
    string's continuation lines are data, not code: prefixing them changes the
    value the cell produces. ``tokenize`` says which lines those are, so a
    triple-quoted literal keeps exactly the content the caller wrote.

    Falls back to a plain indent when the source does not tokenize (incomplete
    input, for instance); the caller reports the resulting error itself.
    """
    lines = code.split("\n")
    try:
        string_ranges: set[int] = set()
        fstring_middle_type = getattr(tokenize, "FSTRING_MIDDLE", None)
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.STRING or (
                fstring_middle_type and token.type == fstring_middle_type
            ):
                start_line = token.start[0] - 1
                end_line = token.end[0] - 1
                if end_line > start_line:
                    # Continuation lines only: the opening line is still code.
                    string_ranges.update(range(start_line + 1, end_line + 1))
    except (tokenize.TokenError, IndentationError):
        return "\n".join(prefix + line if line else line for line in lines)

    return "\n".join(
        line if idx in string_ranges or not line else prefix + line
        for idx, line in enumerate(lines)
    )


_NEVER_GLOBAL = frozenset({"self", "asyncio", "__builtins__"})


class _ModuleScopeBinder(ast.NodeVisitor):
    """Collect names the cell binds in *its own* scope.

    ``ast.walk`` is wrong here: it descends into comprehensions and nested
    function bodies, whose bindings are local to those scopes. Declaring such a
    name ``global`` would both miss the real binding and pull the outer value
    into ``captured_locals`` — the leak that assignment-driven globals exist to
    prevent. So this visitor stops at every new scope.

    ``Del`` counts as a binding: ``del x`` compiles ``x`` as a local unless it
    is declared global, so without it a cell that deletes a session variable
    raises UnboundLocalError before the delete runs.
    """

    def __init__(self) -> None:
        self.names: set[str] = set()

    # --- scopes that do NOT introduce a new binding scope for our purposes ---
    def visit_Module(self, node: ast.Module) -> None:
        for child in node.body:
            self.visit(child)

    def visit_If(self, node: ast.If) -> None:
        # ``test`` can hold a walrus, which binds in this scope.
        self.visit(node.test)
        for child in (*node.body, *node.orelse):
            self.visit(child)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for child in (*node.body, *node.orelse):
            self.visit(child)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        for child in (*node.body, *node.orelse):
            self.visit(child)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        for child in node.body:
            self.visit(child)

    visit_AsyncWith = visit_With

    def visit_Try(self, node: ast.Try) -> None:
        for child in (*node.body, *node.orelse, *node.finalbody):
            self.visit(child)
        for handler in node.handlers:
            # ``except E as err`` binds ``err`` only inside the handler, and
            # Python deletes it afterwards; it is never session state.
            if handler.type is not None:
                self.visit(handler.type)
            for child in handler.body:
                self.visit(child)

    def visit_TryStar(self, node: Any) -> None:
        self.visit_Try(node)

    # --- statements that bind a module-level name ---------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind_target(node.target)
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind_target(node.target)
        self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.names.add(target.id)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # ``:=`` binds in the enclosing scope, which here is the cell's.
        self._bind_target(node.target)
        self.visit(node.value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.names.add(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:
        self.names.update(node.names)

    # --- scopes that DO start a new binding scope: do not descend -----------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ListComp(self, node: ast.ListComp) -> None:
        pass

    def visit_SetComp(self, node: ast.SetComp) -> None:
        pass

    def visit_DictComp(self, node: ast.DictComp) -> None:
        pass

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        pass

    # --- helpers -----------------------------------------------------------
    def _bind_target(self, target: ast.expr) -> None:
        """Record every Name a (possibly destructuring) target binds."""
        if isinstance(target, ast.Name):
            self.names.add(target.id)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)
        elif isinstance(target, ast.Tuple | ast.List):
            for element in target.elts:
                self._bind_target(element)
        # Attribute and Subscript targets mutate an existing object; they bind
        # no new name and must not be declared global.


def _assigned_names(tree: ast.Module) -> set[str]:
    """Names this cell binds in its own scope.

    A name needs ``global`` only when the cell *binds* it: that is what makes
    ``x = x + 5`` read the previous value instead of raising UnboundLocalError,
    and what lets ``del x`` reach the surrounding namespace. Scanning the cell —
    rather than the namespace — keeps the declaration list proportional to what
    the cell touches. Declaring every name in the namespace instead would make
    the wrapper's ``finally`` capture the agent module's entire global scope
    into ``captured_locals`` on every cell.
    """
    binder = _ModuleScopeBinder()
    binder.visit(tree)
    return binder.names


def _global_declarations(tree: ast.Module, bindings: Mapping[str, Any]) -> tuple[str, ...]:
    """Names to declare ``global`` in the wrapper.

    Intersection of two sets: what the cell assigns, and what the namespace
    already holds as plain data. Both halves earn their place — the first keeps
    the list small, the second avoids declaring a name the cell would then have
    to create from nothing.

    Callables and modules are excluded: they are re-imported or redefined rather
    than accumulated, and capturing them would put functions into session state.
    """
    assigned = _assigned_names(tree)
    return tuple(
        name
        for name, value in sorted(bindings.items())
        if name in assigned
        and not name.startswith("_")
        and name not in _NEVER_GLOBAL
        and not callable(value)
        and not isinstance(value, types.ModuleType)
    )


@dataclass(frozen=True)
class PersistedCell:
    """What :meth:`PreparedCell.persist` hands back after a cell ran."""

    captured_locals: dict[str, Any]
    defined_methods: dict[str, Any]


@dataclass(frozen=True)
class PreparedCell:
    """One cell, parsed and rewritten into the async wrapper, ready to run.

    This is the seam both execution backends share. It owns the parts that must
    not diverge — the implicit-return rewrite, the ``global`` declarations, the
    wrapper text, the ``linecache`` entry, and the re-execution of function
    definitions so helpers persist — and nothing else.

    What it deliberately does *not* own, because the adapters genuinely differ:

    * **stream capture** — the in-process adapter uses contextvar-backed
      wrappers so concurrent cells do not interleave; the worker runs one cell
      per process and can redirect ``sys.stdout`` directly.
    * **timeouts** — the in-process adapter uses ``asyncio.wait_for``; the
      sandbox enforces a hard kill in the parent.
    * **persistence** — the worker keeps one long-lived namespace; the
      in-process adapter re-seeds a fresh one per call from ``captured_locals``.

    ``filename`` is also the ``linecache`` key, so tracebacks show the cell's
    own source. ``line_offset`` is the number of wrapper lines before the
    user's first line: error formatting subtracts it.
    """

    code: str
    """The cell source after the implicit-return rewrite."""

    original_source: str
    """The cell source as the caller wrote it, used for the linecache entry."""

    tree: ast.Module
    """The AST of the original source, before the rewrite."""

    filename: str
    line_offset: int
    has_explicit_return: bool
    implicit_return_added: bool
    global_names: tuple[str, ...]
    method_sources: dict[str, str]

    def run(self, namespace: dict[str, Any]) -> Awaitable[Any]:
        """Install the wrapper in ``namespace`` and return its coroutine.

        Awaiting the result runs the cell. The caller owns the event-loop policy
        (timeout, cancellation), the stream capture, and any post-processing of
        ``namespace``.
        """
        exec(compile(self.code, self.filename, "exec"), namespace)
        return namespace["__repl_wrapper__"]()

    def persist(self, namespace: dict[str, Any], *, succeeded: bool) -> PersistedCell:
        """Fold the cell's locals back into ``namespace`` and bind its helpers.

        Must run after :meth:`run` completes, including when the cell raised or
        returned via signal: the wrapper captures locals in a ``finally``.

        ``succeeded`` says whether the cell ran to completion. It gates the two
        things that would otherwise rewrite state a failed cell should not
        touch:

        * Helpers are re-executed from the cell AST so later cells get
          source-relative line numbers. A failed cell's helpers stay defined
          only if the wrapper already bound them, matching the in-process
          adapter, which reports no ``defined_methods`` for a failed cell.
        * The ``linecache`` entry is switched to the original source so helper
          tracebacks line up. Doing that for a *failed* cell would make the
          failing frame's own source line unfindable, because the traceback
          points at wrapper coordinates while the cache would hold source
          coordinates.

        Both callers get the same two views. The worker keeps them in the child
        and reports empty ones upward (live objects cannot cross the process
        boundary); the in-process adapter seeds its next call's globals with
        ``captured_locals`` and reports ``defined_methods`` as feedback.
        """
        # Read, do not pop: a cell that exits via ``return_result()`` raises
        # through this frame, and the caller's own error path still reads the
        # key to build its result. Removing it here emptied ``captured_locals``
        # on exactly the signal path.
        raw = namespace.get("__repl_captured_locals__", {})
        # Every captured name is session state. Bound methods included: a cell
        # may legitimately do ``f = self.some_tool`` and call ``f()`` later, and
        # in the worker this namespace *is* the state.
        captured_locals = dict(raw)
        for name, value in captured_locals.items():
            namespace[name] = value

        func_defs = [
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        if func_defs and succeeded:
            # Helpers are compiled from the cell AST, not through the wrapper, so
            # their line numbers are source-relative and later-cell tracebacks
            # point at the helper's own source.
            #
            # This re-runs the cell's decorators. If one of them raises, that
            # error must not replace the cell's own outcome — ``persist`` runs in
            # a ``finally``, so an exception here would mask a ``return_result``
            # signal, a ``SystemExit`` message, or a cancellation. The cell's
            # result is already decided; a helper that failed to rebind is a
            # missing name, which the next cell reports on its own.
            try:
                exec(
                    compile(ast.Module(body=func_defs, type_ignores=[]), self.filename, "exec"),
                    namespace,
                )
            except Exception:  # noqa: BLE001 - never mask the cell's own outcome
                logger.debug("helper re-execution failed during persist", exc_info=True)
            else:
                # The direct-compiled helpers carry source-relative line numbers;
                # point the cache at the original source so their tracebacks match.
                linecache.cache[self.filename] = (
                    len(self.original_source),
                    None,
                    self.original_source.splitlines(keepends=True),
                    self.filename,
                )
                # The wrapper captured the wrapped function objects; persist the
                # direct-compiled versions so later calls use source-relative lines.
                for node in func_defs:
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                        if node.name in namespace:
                            captured_locals[node.name] = namespace[node.name]

        for name, source in self.method_sources.items():
            fn = namespace.get(name)
            if callable(fn):
                with contextlib.suppress(AttributeError, TypeError):
                    cast(Any, fn)._generated_source = source

        defined_methods = {
            name: namespace[name] for name in self.method_sources if callable(namespace.get(name))
        }
        return PersistedCell(captured_locals=captured_locals, defined_methods=defined_methods)


def prepare_cell(
    code: str,
    *,
    execution_count: int = 1,
    existing_bindings: Mapping[str, Any] | None = None,
) -> PreparedCell:
    """Parse ``code`` and rewrite it into the async REPL wrapper.

    ``existing_bindings`` is the namespace this cell will run in. Each public,
    non-callable, non-module value in it is declared ``global`` so a rebind like
    ``x = x + 5`` reads the previous value instead of raising
    ``UnboundLocalError``. Pass everything the cell can see: in process that is
    ``exec_globals`` merged with the strategy builtins; in the worker it is the
    namespace itself.

    Raises ``SyntaxError`` from :func:`ast.parse`; callers turn that into their
    own error result.
    """
    filename = f"Cell In[{execution_count}]"
    original_source = code
    tree = ast.parse(code, filename=filename)

    method_sources: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.args.args and node.args.args[0].arg == "self":
                segment = ast.get_source_segment(code, node)
                if segment:
                    method_sources[node.name] = segment

    has_explicit_return = _has_top_level_return(tree)

    # Implicit last-expression return (IPython/Jupyter style). Insert ``return ``
    # at the statement's COLUMN, not the line start: a ``;``-compound final line
    # (``a(); b()``) must keep every earlier statement live — line-prefixing made
    # them dead code (or a bogus SyntaxError for ``x = 1; f(x)``).
    implicit_return_added = False
    if tree.body and not has_explicit_return and isinstance(tree.body[-1], ast.Expr):
        last_stmt = tree.body[-1]
        last_line_no = last_stmt.lineno
        lines = code.split("\n")
        if 1 <= last_line_no <= len(lines):
            original = lines[last_line_no - 1]
            # ``col_offset`` is a UTF-8 *byte* offset, while ``original`` is a
            # str indexed by characters. Slicing the str directly shifts the
            # insertion point on any line with non-ASCII text before the last
            # statement — ``d = {'café': 1}; d['café']`` became
            # ``...; dreturn ['café']``. Round-trip through bytes to agree.
            prefix = original.encode("utf-8")[: last_stmt.col_offset].decode("utf-8")
            lines[last_line_no - 1] = f"{prefix}return {original[len(prefix) :]}"
            code = "\n".join(lines)
            implicit_return_added = True

    global_names = _global_declarations(tree, existing_bindings or {})
    global_decl = f"    global {', '.join(global_names)}\n" if global_names else ""
    header = f"async def __repl_wrapper__():\n{global_decl}    try:\n"
    line_offset = header.count("\n")

    indented = _indent(code, "        ")
    wrapper = f"""async def __repl_wrapper__():
{global_decl}    try:
{indented}
    finally:
        __repl_captured_locals__.update({{
            k: v for k, v in locals().items()
            if not k.startswith('_') and k not in ('self', 'asyncio')
        }})
        for _gvar in {set(global_names)!r}:
            if _gvar in globals():
                __repl_captured_locals__[_gvar] = globals()[_gvar]
"""
    linecache.cache[filename] = (
        len(wrapper),
        None,
        wrapper.splitlines(keepends=True),
        filename,
    )

    return PreparedCell(
        code=wrapper,
        original_source=original_source,
        tree=tree,
        filename=filename,
        line_offset=line_offset,
        has_explicit_return=has_explicit_return,
        implicit_return_added=implicit_return_added,
        global_names=global_names,
        method_sources=method_sources,
    )


async def run_cell_source(
    code: str,
    namespace: dict[str, Any],
    *,
    execution_count: int = 1,
    stdout_hook: Callable[[str], None] | None = None,
    stderr_hook: Callable[[str], None] | None = None,
) -> ExecutionResult:
    """Execute ``code`` in ``namespace`` with REPL semantics; never raises.

    Returns an :class:`ExecutionResult`. ``ExecutionSignal`` (``return_result``)
    is captured into ``result.signal``; every other exception into
    ``result.error``. ``namespace`` is mutated in place so definitions persist
    across cells (the worker's long-lived REPL state).
    """
    from nooa.runtime.media_capture import _media_buffer_var, _MediaBuffer

    stdout_buf = _HookedBuffer(stdout_hook)
    stderr_buf = _HookedBuffer(stderr_hook)
    media_buffer = _MediaBuffer(max_attachments=100)
    media_token = _media_buffer_var.set(media_buffer)

    returned_value: Any = _NO_RETURN
    has_explicit_return = False
    wrapper_line_offset = 0

    def build_result(*, error=None, signal=None) -> ExecutionResult:
        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            error=error,
            signal=signal,
            defined_methods=defined_methods,
            returned_value=returned_value,
            explicit_return=has_explicit_return,
            captured_locals={},
            images=media_buffer.blocks,
            wrapper_line_offset=wrapper_line_offset,
        )

    defined_methods: dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                cell = prepare_cell(
                    code,
                    execution_count=execution_count,
                    existing_bindings=namespace,
                )
            except SyntaxError as exc:
                return build_result(error=exc)

            has_explicit_return = cell.has_explicit_return
            wrapper_line_offset = cell.line_offset

            # The wrapper's finally block writes into this key; it must exist
            # before the cell runs so a raise cannot leave it unbound.
            namespace["__repl_captured_locals__"] = {}
            succeeded = False
            try:
                result_value = await cell.run(namespace)
                succeeded = True
            finally:
                # Keep REPL state even when the cell raises or returns via signal.
                defined_methods = cell.persist(namespace, succeeded=succeeded).defined_methods
                # persist() reads the key rather than popping it (the caller's
                # error path may still need it), so the worker clears it here:
                # its namespace outlives the cell.
                namespace.pop("__repl_captured_locals__", None)

            if has_explicit_return:
                returned_value = result_value
            elif cell.implicit_return_added and result_value is not None:
                returned_value = result_value

            return build_result()
    except ExecutionSignal as sig:
        return build_result(signal=sig)
    except (SystemExit, KeyboardInterrupt) as exc:
        wrapped = RuntimeError(
            f"{type(exc).__name__} raised inside generated code. Use return_result() "
            "or a normal return to finish a cell, not sys.exit()/exit()/quit()."
        )
        wrapped.__cause__ = exc
        return build_result(error=wrapped)
    except Exception as exc:
        return build_result(error=exc)
    finally:
        _media_buffer_var.reset(media_token)
