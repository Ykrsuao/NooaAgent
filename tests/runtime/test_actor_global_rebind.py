# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell may rebind a name that is already in the agent module's globals.

The REPL wrapper declares session variables ``global`` so ``x = x + 5`` works
when ``x`` came from a previous cell. It built that list from the strategy's
``builtins`` only, ignoring ``exec_globals`` -- which already holds the agent
module's own module-level names. Rebinding one of those made Python treat it as
a local, so the read side raised ``UnboundLocalError``:

    MODULE_CONST = MODULE_CONST + 1   # UnboundLocalError: in process
                                      # worked under the sandbox backend

The sandbox declares every non-callable namespace name, so the two backends
disagreed. Both sources now feed the list.
"""

from __future__ import annotations

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.unifiedllm import FakeLLMClient

MODULE_CONST = 100
"""Module-level, non-callable: the shape that used to fail."""


class _ModuleConstAgent(Agent, llm=FakeLLMClient()):
    pass


@pytest.mark.asyncio
async def test_rebinding_a_module_constant_reads_its_previous_value():
    runtime = ActorRuntime(_ModuleConstAgent())
    result = await runtime.execute_code(
        "MODULE_CONST = MODULE_CONST + 1\nMODULE_CONST",
        wrap_in_function=True,
        validate=False,
    )
    assert result.error is None, f"cell errored: {result.error}"
    assert result.returned_value == 101


@pytest.mark.asyncio
async def test_rebinding_a_module_constant_does_not_leak_into_the_module():
    """The rebind lands in the cell's globals, not the importing module's."""
    runtime = ActorRuntime(_ModuleConstAgent())
    await runtime.execute_code(
        "MODULE_CONST = MODULE_CONST + 1", wrap_in_function=True, validate=False
    )
    assert MODULE_CONST == 100


@pytest.mark.asyncio
async def test_rebinding_a_module_list_mutates_it_in_place():
    runtime = ActorRuntime(_ModuleConstAgent())
    result = await runtime.execute_code(
        "MODULE_LIST = MODULE_LIST + [3]\nreturn MODULE_LIST",
        wrap_in_function=True,
        validate=False,
    )
    assert result.error is None, f"cell errored: {result.error}"
    assert result.returned_value == [1, 2, 3]


MODULE_LIST = [1, 2]


MODULE_BIG = list(range(1000))
"""Module-level data the cell below never touches."""


@pytest.mark.asyncio
async def test_untouched_module_names_are_not_captured():
    """Only names the cell assigns may be declared global.

    The wrapper's ``finally`` captures every ``global`` name into
    ``captured_locals``, and strategies fold that into the session. Declaring
    the whole namespace therefore copied the agent module's entire global scope
    into session state on every cell — a leak that grew with the cell count.
    """
    runtime = ActorRuntime(_ModuleConstAgent())
    result = await runtime.execute_code("1 + 1", wrap_in_function=True, validate=False)

    assert result.error is None
    assert "MODULE_BIG" not in result.captured_locals
    assert "MODULE_CONST" not in result.captured_locals


@pytest.mark.asyncio
async def test_assigned_module_names_are_still_captured():
    """The control: a name the cell does assign is captured, so rebinds persist."""
    runtime = ActorRuntime(_ModuleConstAgent())
    result = await runtime.execute_code(
        "MODULE_CONST = MODULE_CONST + 1", wrap_in_function=True, validate=False
    )

    assert result.error is None
    assert result.captured_locals["MODULE_CONST"] == 101


@pytest.mark.asyncio
async def test_callable_module_names_are_not_declared_global():
    """Helpers and imported callables stay ordinary names; only data is rebound."""
    runtime = ActorRuntime(_ModuleConstAgent())
    result = await runtime.execute_code(
        "def MODULE_CONST():\n    return 'shadowed'\nMODULE_CONST()",
        wrap_in_function=True,
        validate=False,
    )
    assert result.error is None, f"cell errored: {result.error}"
    assert result.returned_value == "shadowed"
