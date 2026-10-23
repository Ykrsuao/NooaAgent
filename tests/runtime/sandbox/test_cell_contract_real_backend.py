# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both execution backends agree when driven through the real sandbox worker.

``tests/runtime/test_cell_contract.py`` compares ``ActorRuntime.execute_code``
against ``run_cell_source`` in this process. Since ``prepare_cell`` became the
shared seam, those two share their transform step, so that file can no longer
catch a divergence *inside* the transform — both sides would be wrong together.

This file closes that gap by driving ``SandboxedExecutor``, which crosses the
parts the in-process comparison never touches: pickling the cell, the worker
process, the wire codec, and the parent-side decode. It is marked ``sandbox``
because it needs a real worker (fork on Unix, spawn on Windows) and is run by
the dedicated CI job.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.unifiedllm import FakeLLMClient

pytestmark = pytest.mark.sandbox


class _RecorderAgent(Agent, llm=FakeLLMClient()):
    """Module-level so ``start_method='spawn'`` can import it by name."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[str] = []

    def record(self, tag: str) -> str:
        self.calls.append(tag)
        return f"recorded:{tag}"


# The cases most likely to diverge: the two rewrites prepare_cell performs, and
# the value shapes that the wire codec has to carry back.
CORPUS: list[tuple[str, str, str]] = [
    ("implicit_int", "1 + 1", "2"),
    ("explicit", "return 42", "42"),
    ("compound_final_line", "a = 1; a + 1", "2"),
    ("multiline_str", 's = """\nhello\n"""\nreturn s', "'\\nhello\\n'"),
    ("multiline_str_implicit", 's = """\nhello\n"""\ns', "'\\nhello\\n'"),
    ("multiline_fstring", 'name = "x"\nf"""\n{name}\n"""', "'\\nx\\n'"),
    ("helper_in_multiline", 'def f():\n    return """\n  y\n  """\nreturn f()', "'\\n  y\\n  '"),
    ("helper_def", "def helper(n):\n    return n * 2\nhelper(3)", "6"),
    ("list_comp", "[i * 2 for i in range(3)]", "[0, 2, 4]"),
    ("dict_literal", '{"a": 1, "b": 2}', "{'a': 1, 'b': 2}"),
    ("self_call", "self.record('a')", "'recorded:a'"),
]


# Windows has no "fork"; the cell semantics under test are identical either way.
_START_METHOD = "spawn" if os.name == "nt" else "fork"


def _sandbox_config() -> SandboxConfig:
    # require=False so a CI host without a guard mechanism still runs the cell;
    # network on so the worker's own imports behave like the in-process ones.
    return SandboxConfig(require=False, network=True, filesystem=False, start_method=_START_METHOD)


@pytest.fixture(scope="module")
async def executor():
    """One worker for the whole module.

    Starting a container worker re-syncs the runtime root and waits for a ready
    handshake — tens of seconds. Per-test fixtures would pay that once per case
    and dominate the suite's runtime.
    """
    ex = SandboxedExecutor(
        _RecorderAgent(),
        _sandbox_config(),
        cell_timeout=30.0,
        framework_builtins={},
    )
    try:
        yield ex
    finally:
        await ex.aclose()


@pytest.mark.parametrize("name,code,expected", CORPUS, ids=[c[0] for c in CORPUS])
async def test_sandbox_matches_inprocess(name: str, code: str, expected: str, executor) -> None:
    """The worker's observable result must equal the in-process one."""
    inprocess = await ActorRuntime(_RecorderAgent()).execute_code(
        code, wrap_in_function=True, validate=False
    )
    assert inprocess.error is None, f"in-process errored: {inprocess.error}"
    assert repr(inprocess.returned_value) == expected

    sandbox = await executor.run_cell(code, execution_count=1)
    assert sandbox.error is None, f"sandbox errored: {sandbox.error}"
    assert repr(sandbox.returned_value) == expected, f"{name}: sandbox and in-process disagree"
    assert sandbox.explicit_return == inprocess.explicit_return
    assert sandbox.stdout == inprocess.stdout


async def test_sandbox_preserves_namespace_across_cells(executor) -> None:
    """The worker's namespace is long-lived; state written in one cell is readable."""
    first = await executor.run_cell("kept = [1]", execution_count=1)
    assert first.error is None

    second = await executor.run_cell("return kept + [2]", execution_count=2)
    assert second.error is None
    assert second.returned_value == [1, 2]
