# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions found by reviewing the prepare_cell extraction.

Each test here failed against the extraction as first written and passes now.
They are kept together because they share a cause: the shared seam moved work
into ``persist()``, which runs in a ``finally`` and therefore sees cells that
did *not* complete.
"""

from __future__ import annotations

import ast

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.cell_core import _assigned_names, run_cell_source
from nooa.unifiedllm import FakeLLMClient


class _Agent(Agent, llm=FakeLLMClient()):
    pass


class _Tool:
    def value(self) -> int:
        return 7


# --- persist() must never mask the cell's own outcome ------------------------


async def test_a_failing_decorator_does_not_replace_the_cell_error():
    """persist re-executes helpers; a decorator raising there must not win.

    The cell's own ValueError is the answer the model needs. Re-binding helpers
    is bookkeeping, so a failure in it is swallowed.
    """
    code = "raise ValueError('ORIGINAL')\n@missing_decorator\ndef late():\n    return 1\n"
    result = await run_cell_source(code, {}, execution_count=1)

    assert type(result.error).__name__ == "ValueError"
    assert "ORIGINAL" in str(result.error)


async def test_a_failing_decorator_does_not_swallow_return_result():
    """The signal path matters most: return_result must survive."""
    from nooa.events import ExecutionSignal

    def return_result(value=None):
        raise ExecutionSignal("return_result", {"result": value})

    code = "return_result(5)\n@missing_decorator\ndef late():\n    return 1\n"
    result = await run_cell_source(code, {"return_result": return_result}, execution_count=1)

    assert result.error is None, result.error
    assert result.signal is not None


async def test_inprocess_failing_decorator_keeps_the_real_error():
    runtime = ActorRuntime(_Agent())
    code = "raise ValueError('ORIGINAL')\n@missing_decorator\ndef late():\n    return 1\n"
    result = await runtime.execute_code(code, wrap_in_function=True, validate=False)

    assert type(result.error).__name__ == "ValueError"


# --- a failed cell must not leave helpers behind -----------------------------


async def test_failed_cell_does_not_bind_its_helpers():
    """A cell that raised never completed, so its defs are not session state."""
    ns: dict = {}
    await run_cell_source(
        "raise ValueError('x')\ndef late():\n    return 1\n", ns, execution_count=1
    )

    assert "late" not in ns


async def test_successful_cell_still_binds_its_helpers():
    """The control: a cell that completed does bind them."""
    ns: dict = {}
    await run_cell_source("def helper():\n    return 1\nhelper()", ns, execution_count=1)

    assert callable(ns.get("helper"))


# --- the failing frame keeps its source line --------------------------------


async def test_traceback_keeps_source_when_the_cell_defines_a_helper():
    """Switching linecache to the original source must not happen on failure.

    The traceback points at wrapper coordinates; if the cache held source
    coordinates, the failing line's text would be unfindable and the model would
    get a traceback with no source at all.
    """
    from nooa.errors.formatting import format_error_for_llm

    code = "def helper():\n    return 1\nraise ValueError('boom')"
    result = await ActorRuntime(_Agent()).execute_code(code, wrap_in_function=True, validate=False)
    rendered = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)

    assert "raise ValueError('boom')" in rendered, rendered


# --- the worker's namespace is its state ------------------------------------


async def test_bound_method_alias_persists_in_the_worker_namespace():
    """``f = self.tool`` then ``f()`` is a normal LLM pattern."""
    ns: dict = {"self": _Tool()}
    await run_cell_source("f = self.value", ns, execution_count=1)

    assert "f" in ns
    result = await run_cell_source("f()", ns, execution_count=2)
    assert result.error is None, result.error
    assert result.returned_value == 7


# --- binding forms the global list must recognise ---------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("x = 1", {"x"}),
        ("x += 1", {"x"}),
        ("del x", {"x"}),
        ("import os as x", {"x"}),
        ("from os import path as x", {"x"}),
        ("for x in range(3): pass", {"x"}),
        ("with open('f') as x: pass", {"x"}),
        ("a, *x = [1, 2]", {"a", "x"}),
        ("if (x := 1): pass", {"x"}),
        # Not bindings in the cell's own scope:
        ("[x for x in range(3)]", set()),
        ("{x: 1 for x in range(3)}", set()),
        ("def f():\n    x = 1", set()),
        ("class C:\n    x = 1", set()),
        ("try:\n    pass\nexcept ValueError as x:\n    pass", set()),
        ("obj.x = 1", set()),
        ("d['x'] = 1", set()),
    ],
)
def test_assigned_names_recognises_each_binding_form(code: str, expected: set[str]) -> None:
    assert _assigned_names(ast.parse(code)) == expected


async def test_deleting_a_session_variable_reaches_the_namespace():
    """``del`` is a binding: without a global declaration it raises first."""
    runtime = ActorRuntime(_Agent())
    first = await runtime.execute_code("gone = 1", wrap_in_function=True, validate=False)
    result = await runtime.execute_code(
        "del gone\nreturn 'ok'",
        wrap_in_function=True,
        validate=False,
        builtins=dict(first.captured_locals),
    )

    assert result.error is None, result.error
    assert result.returned_value == "ok"


# --- the implicit-return rewrite is byte-correct on non-ASCII lines ---------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("d = {'café': 1}; d['café']", 1),
        ("print('你好'); result = 1; result", 1),
        ("s = '日本語'; len(s)", 3),
        ("emoji = '🎉'; emoji", "🎉"),
    ],
)
async def test_implicit_return_handles_non_ascii_before_the_last_statement(
    code: str, expected: object
) -> None:
    """``col_offset`` is a UTF-8 byte offset; the source is indexed by character."""
    result = await run_cell_source(code, {}, execution_count=1)

    assert result.error is None, result.error
    assert result.returned_value == expected
