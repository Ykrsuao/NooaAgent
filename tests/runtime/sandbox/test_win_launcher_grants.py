# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which paths a container start grants, computed without touching an ACL."""

from __future__ import annotations

import sys

import pytest

from nooa.runtime.sandbox.config import PathGrant, ResolvedSpec
from nooa.runtime.sandbox.errors import SandboxUnavailable
from nooa.runtime.sandbox.win_runtime import RuntimeRoot

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the launcher binds Win32 APIs")


@pytest.fixture
def layout(tmp_path):
    root = RuntimeRoot(tmp_path / "rt")
    root.tokenizer_cache.mkdir(parents=True)
    (root.tokenizer_cache / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4").write_bytes(b"")
    workspace = tmp_path / "ws"
    data = tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    return root, workspace, data


def test_workspace_is_granted_once_last_and_with_modify(layout):
    from nooa.runtime.sandbox.win_launcher import MODIFY, READ_EXECUTE, _grant_plan

    root, workspace, data = layout
    spec = ResolvedSpec(
        filesystem=True,
        path_grants=(
            PathGrant(str(workspace), write=True, required=True),
            PathGrant(str(data), write=False, required=True),
            PathGrant(r"C:\Windows", write=False),  # a system path: not granted
        ),
    )
    # The workspace is also an import root: it must still end up MODIFY.
    plan = _grant_plan(spec, root, str(workspace), [str(workspace), str(data)])
    paths = [path for path, _ in plan]

    assert paths.count(str(workspace)) == 1
    assert plan[-1] == (str(workspace), MODIFY)
    assert paths.count(str(data)) == 1
    assert dict(plan)[str(data)] == READ_EXECUTE
    assert dict(plan)[str(root.root)] == READ_EXECUTE
    assert r"C:\Windows" not in paths


def test_a_missing_required_path_still_refuses_the_start(layout, tmp_path):
    from nooa.runtime.sandbox.win_launcher import _grant_plan

    root, workspace, _ = layout
    missing = tmp_path / "gone"
    spec = ResolvedSpec(path_grants=(PathGrant(str(missing), write=True, required=True),))
    with pytest.raises(SandboxUnavailable, match="does not exist"):
        _grant_plan(spec, root, str(missing), [])
