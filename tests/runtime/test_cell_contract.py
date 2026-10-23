# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterise the cell-execution contract that both backends must honour.

NOOA runs generated Python through two adapters: ``ActorRuntime.execute_code``
in the agent's own process, and ``run_cell_source`` inside the sandbox worker.
They implement the same REPL semantics, and that sameness is the whole promise
of ``execution_backend`` -- switching backends must not change what a cell does.

Nothing enforced that promise. The two implementations drifted: the sandbox
copy indented multiline string contents into the cell's value, so the same
triple-quoted literal produced different data per backend. An example-based
suite missed it because no test combined a multiline string with the wrapper.

This file is the net. ``CORPUS`` is run through both adapters and compared
field by field on everything a caller can observe. The in-process results are
also pinned as literals, so a refactor of either side fails here instead of in
production.
"""

from __future__ import annotations

from typing import Any

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.cell_core import run_cell_source
from nooa.unifiedllm import FakeLLMClient


class _RecorderAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def record(self, tag: str) -> str:
        self.calls.append(tag)
        return f"recorded:{tag}"


# (name, code, expected returned value as repr, expected calls)
# ``returned`` is the ``repr`` so a str and its escape sequences are pinned too.
CORPUS: list[tuple[str, str, str, list[str]]] = [
    ("implicit_int", "1 + 1", "2", []),
    ("implicit_str", "'abc'", "'abc'", []),
    ("implicit_assignment_only", "x = 1", "<object object at", []),
    ("explicit", "return 42", "42", []),
    ("explicit_none", "return None", "None", []),
    ("compound_final_line", "a = 1; a + 1", "2", []),
    ("multiline_str_explicit", 's = """\nhello\n"""\nreturn s', "'\\nhello\\n'", []),
    ("multiline_str_implicit", 's = """\nhello\n"""\ns', "'\\nhello\\n'", []),
    ("multiline_fstring", 'name = "x"\nf"""\n{name}\n"""', "'\\nx\\n'", []),
    ("helper_def", "def helper(n):\n    return n * 2\nhelper(3)", "6", []),
    ("helper_def_explicit", "def helper(n):\n    return n * 2\nreturn helper(3)", "6", []),
    (
        "nested_def",
        "def outer():\n    def inner():\n        return 5\n    return inner()\nouter()",
        "5",
        [],
    ),
    ("self_call", "self.record('a')", "'recorded:a'", ["a"]),
    ("self_call_twice", "self.record('a'); self.record('b')", "'recorded:b'", ["a", "b"]),
    ("list_comp", "[i * 2 for i in range(3)]", "[0, 2, 4]", []),
    ("dict_literal", '{"a": 1, "b": 2}', "{'a': 1, 'b': 2}", []),
    ("multiline_call", "sum([\n    1,\n    2,\n])", "3", []),
    ("class_def", "class C:\n    pass\nC.__name__", "'C'", []),
    ("rebind_module_const", "CONTRACT_CONST = CONTRACT_CONST + 1\nCONTRACT_CONST", "101", []),
]

CONTRACT_CONST = 100
"""Module-level data the rebind cell above reads and reassigns."""

# Cells whose outcome is an error rather than a value.
ERROR_CORPUS: list[tuple[str, str, str]] = [
    ("raise_value_error", "raise ValueError('boom')", "ValueError"),
    ("syntax_error", "def (", "SyntaxError"),
    ("undefined_name", "nope", "NameError"),
]


def _comparable(result: Any) -> dict[str, Any]:
    """The fields both backends must agree on.

    Three exclusions, each for a stated reason:

    * ``captured_locals`` and ``defined_methods`` hold live objects the worker
      cannot send across the process boundary, so they are documented as empty
      on the parent side (see ``runtime/sandbox/serialization.py``).
    * ``wrapper_line_offset`` is not a backend constant. It counts the wrapper
      header lines for *this* call, and the header grows by one line whenever a
      session variable needs a ``global`` declaration. Two runs of the same cell
      legitimately differ when their namespaces do. It is checked for internal
      consistency in ``test_wrapper_offset_matches_the_emitted_header``.
    """
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returned": repr(result.returned_value),
        "explicit_return": result.explicit_return,
        "error": type(result.error).__name__ if result.error else None,
    }


@pytest.mark.asyncio
async def test_error_line_numbers_point_at_the_user_source() -> None:
    """Error formatting subtracts the offset, so the user must see their own line.

    Checked end to end rather than against the other backend: the header grows a
    line when a session variable needs a ``global`` declaration, so the raw
    offset legitimately varies. What must not vary is the line the user is told
    about.
    """
    from nooa.errors.formatting import format_error_for_llm

    # The raise sits on line 2 of the cell, in both adapters.
    code = "x = 1\nraise ValueError('boom')"

    inprocess = await ActorRuntime(_RecorderAgent()).execute_code(
        code, wrap_in_function=True, validate=False
    )
    sandbox = await run_cell_source(code, {"self": _RecorderAgent()}, execution_count=1)

    for result in (inprocess, sandbox):
        rendered = format_error_for_llm(result.error, code, line_offset=result.wrapper_line_offset)
        assert "line 2" in rendered, rendered
        assert "line 1" not in rendered, rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("name,code,expected,expected_calls", CORPUS, ids=[c[0] for c in CORPUS])
async def test_inprocess_behaviour_is_pinned(
    name: str, code: str, expected: str, expected_calls: list[str]
) -> None:
    """The in-process adapter's observable behaviour, pinned as literals."""
    agent = _RecorderAgent()
    result = await ActorRuntime(agent).execute_code(code, wrap_in_function=True, validate=False)

    assert result.error is None, f"cell errored: {result.error}"
    returned = repr(result.returned_value)
    if expected == "<object object at":
        # The no-return sentinel is an identity object; only its kind is stable.
        assert returned.startswith(expected)
    else:
        assert returned == expected
    assert agent.calls == expected_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name,code,expected,expected_calls", CORPUS, ids=[c[0] for c in CORPUS])
async def test_backends_agree(
    name: str, code: str, expected: str, expected_calls: list[str]
) -> None:
    """Switching ``execution_backend`` must not change what the cell does."""
    inprocess_agent = _RecorderAgent()
    inprocess = await ActorRuntime(inprocess_agent).execute_code(
        code, wrap_in_function=True, validate=False
    )

    sandbox_agent = _RecorderAgent()
    # The worker's namespace is what the parent hands it; give it the same
    # module-level names the in-process adapter finds in exec_globals.
    namespace: dict[str, Any] = {"self": sandbox_agent, "CONTRACT_CONST": CONTRACT_CONST}
    sandbox = await run_cell_source(code, namespace, execution_count=1)

    assert _comparable(sandbox) == _comparable(inprocess)
    assert sandbox_agent.calls == inprocess_agent.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("name,code,error_type", ERROR_CORPUS, ids=[c[0] for c in ERROR_CORPUS])
async def test_backends_agree_on_errors(name: str, code: str, error_type: str) -> None:
    """A failing cell fails the same way under both adapters."""
    inprocess = await ActorRuntime(_RecorderAgent()).execute_code(
        code, wrap_in_function=True, validate=False
    )
    sandbox = await run_cell_source(code, {"self": _RecorderAgent()}, execution_count=1)

    assert type(inprocess.error).__name__ == error_type
    assert type(sandbox.error).__name__ == error_type
    assert _comparable(sandbox) == _comparable(inprocess)
