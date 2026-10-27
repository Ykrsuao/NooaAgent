# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a worker turns the parent's ``attr`` answer into what a cell receives.

The parent answers a callable with a flag and ``result=None``: the callable
itself never crosses. Every worker path must honour that flag. The fork path
read only ``proxy`` for an attribute its inherited copy lacked, so a tool the
parent gained after the fork resolved to ``None`` in the cell.

The answers here come from the parent's real ``_dispatch_tool_call``, so a
change to either side of the contract fails these tests.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.worker import _NestedProxy, _resolve_child_attr


class _LiveAgent:
    """The parent's agent, as it is now."""

    limit = 3

    def __init__(self) -> None:
        self.memory = _Unpicklable()

    def helper(self) -> str:
        return "ran on the parent"

    async def fetch(self) -> str:
        return "fetched on the parent"


class _Unpicklable:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def recall(self) -> str:
        return "recalled"


class _ForkedCopy:
    """What the worker inherited: taken before ``helper``/``fetch`` existed,
    and when ``limit`` was still a plain value."""

    limit = 3


class _Broker:
    """Answers with the parent's real dispatch, as ``ChildBroker`` would."""

    def __init__(self, executor: SandboxedExecutor) -> None:
        self._executor = executor

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        # A worker's broker calls are synchronous and the parent answers on its
        # own loop; a separate thread keeps that true inside an async test.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._executor._dispatch_tool_call(msg)).result()

    def get_attr(self, path: list[str]) -> tuple[Any, bool, bool, bool]:
        r = self._dispatch({"kind": "attr", "path": path})
        assert r["ok"], r
        return (
            r.get("result"),
            bool(r.get("proxy")),
            bool(r.get("callable")),
            bool(r.get("is_async")),
        )

    def call(self, path: list[str], args: tuple, kwargs: dict) -> Any:
        return self.call_result(path, args, kwargs)[0]

    def call_result(self, path: list[str], args: tuple, kwargs: dict) -> tuple[Any, bool]:
        r = self._dispatch({"kind": "call", "path": path, "args": args, "kwargs": kwargs})
        assert r["ok"], r
        return r["result"], bool(r.get("was_async"))


@pytest.fixture
def agent() -> _LiveAgent:
    return _LiveAgent()


@pytest.fixture
def broker(agent: _LiveAgent) -> _Broker:
    # Bypass construction: these tests exercise only the broker contract, not
    # a worker or the host's capability probe.
    executor = object.__new__(SandboxedExecutor)
    executor._agent = agent
    executor._max_error = 2000
    return _Broker(executor)


@pytest.mark.parametrize("local", [_ForkedCopy(), None], ids=["fork", "spawn"])
def test_callable_missing_from_the_local_copy_is_brokered(broker, local):
    helper = _resolve_child_attr(broker, local, [], "helper")
    assert helper is not None
    assert helper() == "ran on the parent"


@pytest.mark.parametrize("local", [_ForkedCopy(), None], ids=["fork", "spawn"])
def test_async_callable_missing_from_the_local_copy_stays_awaitable(broker, local):
    fetch = _resolve_child_attr(broker, local, [], "fetch")

    async def cell() -> str:
        return await fetch()

    assert asyncio.run(cell()) == "fetched on the parent"


def test_value_the_parent_rebound_to_a_callable_is_brokered(broker, agent):
    agent.limit = lambda: 7  # a plain value in the forked copy, callable now
    limit = _resolve_child_attr(broker, _ForkedCopy(), [], "limit")
    assert limit() == 7


def test_plain_values_still_cross_as_values(broker):
    assert _resolve_child_attr(broker, _ForkedCopy(), [], "limit") == 3


def test_unpicklable_attribute_becomes_a_nested_proxy(broker):
    memory = _resolve_child_attr(broker, _ForkedCopy(), [], "memory")
    assert isinstance(memory, _NestedProxy)
    assert memory.recall() == "recalled"
