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
import tokenize
import types
from collections.abc import Callable
from typing import Any, cast

from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal


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
    cell_filename = f"Cell In[{execution_count}]"

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
            source_code = code
            try:
                tree = ast.parse(code, filename=cell_filename)
            except SyntaxError as exc:
                return build_result(error=exc)

            method_sources: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if node.args.args and node.args.args[0].arg == "self":
                        seg = ast.get_source_segment(code, node)
                        if seg:
                            method_sources[node.name] = seg

            has_explicit_return = _has_top_level_return(tree)

            # Implicit last-expression return (IPython/Jupyter style).
            # Insert ``return `` at the statement's COLUMN, not the line start:
            # a ``;``-compound final line (``a(); b()``) must keep every earlier
            # statement live — line-prefixing made them dead code (or a bogus
            # SyntaxError for ``x = 1; f(x)``).
            implicit_return_added = False
            if tree.body and not has_explicit_return and isinstance(tree.body[-1], ast.Expr):
                last_stmt = tree.body[-1]
                last_line_no = last_stmt.lineno
                lines = code.split("\n")
                if 1 <= last_line_no <= len(lines):
                    original = lines[last_line_no - 1]
                    col = last_stmt.col_offset
                    lines[last_line_no - 1] = f"{original[:col]}return {original[col:]}"
                    code = "\n".join(lines)
                    implicit_return_added = True

            # Names to declare ``global`` so REPL rebinds (``x = x + 1``) work.
            global_vars = [
                name
                for name, val in namespace.items()
                if (
                    not name.startswith("_")
                    and name not in ("self", "asyncio", "__builtins__")
                    and not callable(val)
                    and not isinstance(val, types.ModuleType)
                )
            ]
            global_decl = f"    global {', '.join(global_vars)}\n" if global_vars else ""
            global_var_set = set(global_vars)

            wrapper_header = f"async def __repl_wrapper__():\n{global_decl}    try:\n"
            wrapper_line_offset = wrapper_header.count("\n")

            namespace["__repl_captured_locals__"] = {}
            indented = _indent(code, "        ")
            wrapper = f"""async def __repl_wrapper__():
{global_decl}    try:
{indented}
    finally:
        __repl_captured_locals__.update({{
            k: v for k, v in locals().items()
            if not k.startswith('_') and k not in ('self', 'asyncio')
        }})
        for _gvar in {global_var_set!r}:
            if _gvar in globals():
                __repl_captured_locals__[_gvar] = globals()[_gvar]
"""
            linecache.cache[cell_filename] = (
                len(wrapper),
                None,
                wrapper.splitlines(keepends=True),
                cell_filename,
            )

            exec(compile(wrapper, cell_filename, "exec"), namespace)
            try:
                result_value = await namespace["__repl_wrapper__"]()
            finally:
                # Keep REPL state even when the cell raises or returns via signal.
                captured = namespace.pop("__repl_captured_locals__", {})
                for k, v in captured.items():
                    namespace[k] = v

            if has_explicit_return:
                returned_value = result_value
            elif implicit_return_added and result_value is not None:
                returned_value = result_value

            # Re-bind top-level function defs so helpers persist by name.
            func_defs: list[ast.stmt] = [
                n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            if func_defs:
                exec(
                    compile(ast.Module(body=func_defs, type_ignores=[]), cell_filename, "exec"),
                    namespace,
                )
                # Persisted helpers are compiled directly from the cell AST, not
                # through the async wrapper. Keep their source cache aligned with
                # those original line numbers for later-cell tracebacks.
                linecache.cache[cell_filename] = (
                    len(source_code),
                    None,
                    source_code.splitlines(keepends=True),
                    cell_filename,
                )
            for name, src in method_sources.items():
                fn = namespace.get(name)
                if callable(fn):
                    with contextlib.suppress(AttributeError, TypeError):
                        cast(Any, fn)._generated_source = src
            defined_methods = {
                name: namespace[name] for name in method_sources if callable(namespace.get(name))
            }
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
