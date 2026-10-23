# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintenance of the sandbox runtime root, without building a real one.

Every root here is a few placeholder files under ``tmp_path``; nothing runs an
interpreter, touches an ACL or reaches the network.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from nooa._file_lock import lock_file, unlock_file
from nooa.runtime.sandbox import win_runtime
from nooa.runtime.sandbox.errors import SandboxUnavailable
from nooa.runtime.sandbox.win_runtime import RuntimeRoot, build_root, sync_package


@pytest.fixture
def source(tmp_path):
    """A stand-in interpreter prefix and site-packages to copy from."""
    prefix = tmp_path / "src-prefix"
    (prefix / "Lib").mkdir(parents=True)
    (prefix / "python.exe").write_bytes(b"")
    site = tmp_path / "src-site"
    site.mkdir()
    return prefix, site


@pytest.fixture(autouse=True)
def _no_bytecode(monkeypatch):
    monkeypatch.setattr(win_runtime, "compile_bytecode", lambda *a, **k: "skipped")


def test_rebuild_refuses_a_directory_it_did_not_create(tmp_path, source):
    # --root is typed by the user, and the removal is recursive.
    target = tmp_path / "projects"
    target.mkdir()
    (target / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(SandboxUnavailable, match="will not be deleted"):
        build_root(RuntimeRoot(target), interpreter_prefix=source[0], site_packages=source[1])
    assert (target / "keep.txt").read_text(encoding="utf-8") == "user data"


@pytest.mark.parametrize("existing", ["empty", "marked", "synced"])
def test_rebuild_replaces_an_empty_or_recognised_root(tmp_path, source, existing):
    root = RuntimeRoot(tmp_path / "rt")
    root.root.mkdir()
    if existing == "marked":
        (root.root / win_runtime._ROOT_MARKER).write_text("", encoding="utf-8")
        (root.root / "stale.txt").write_text("old build", encoding="utf-8")
    elif existing == "synced":  # a root built before the marker existed
        root.python.write_bytes(b"")
        root.site_packages.mkdir(parents=True)
        (root.site_packages / ".nooa-sync.json").write_text("{}", encoding="utf-8")
        (root.root / "stale.txt").write_text("old build", encoding="utf-8")
    build_root(root, interpreter_prefix=source[0], site_packages=source[1])
    assert not (root.root / "stale.txt").exists()
    assert (root.root / win_runtime._ROOT_MARKER).exists()
    assert root.python.exists()


def test_sync_holds_the_root_lock_while_it_copies(tmp_path, monkeypatch):
    # Two sessions starting after a source edit both saw a stale signature and
    # interleaved their rmtree/copytree on the shared root.
    root = RuntimeRoot(tmp_path / "rt")
    root.root.mkdir()
    seen: list[str] = []

    def probe(root_, *, force, packages):
        fd = os.open(root_.root / win_runtime._SYNC_LOCK, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                lock_file(fd, blocking=False)
        finally:
            os.close(fd)
        seen.append("held")
        return "probed"

    monkeypatch.setattr(win_runtime, "_sync_package_locked", probe)
    assert sync_package(root) == "probed"
    assert seen == ["held"]
    # Released afterwards: the next session can take it.
    fd = os.open(root.root / win_runtime._SYNC_LOCK, os.O_RDWR)
    try:
        lock_file(fd, blocking=False)
        unlock_file(fd)
    finally:
        os.close(fd)


@pytest.fixture
def tokenizer_root(tmp_path):
    root = RuntimeRoot(tmp_path / "rt")
    (root.site_packages / "tiktoken_ext").mkdir(parents=True)
    (root.site_packages / "tiktoken_ext" / "openai_public.py").write_text("v1", encoding="utf-8")
    root.python.write_bytes(b"")
    return root


def _fake_warm_up(calls: list[object]):
    def run(argv, **kwargs):
        calls.append(argv)
        cache = kwargs["env"]["TIKTOKEN_CACHE_DIR"]
        with open(os.path.join(cache, "cl100k"), "wb") as handle:
            handle.write(b"data")
        return subprocess.CompletedProcess(argv, 0, "", "")

    return run


def test_a_warm_cache_is_not_warmed_again(tokenizer_root, monkeypatch):
    # Every container start spawned an interpreter to rebuild a cached encoding.
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", _fake_warm_up(calls))
    assert "file(s)" in win_runtime.warm_tokenizer_cache(tokenizer_root)
    assert "(warm)" in win_runtime.warm_tokenizer_cache(tokenizer_root)
    assert len(calls) == 1


def test_a_changed_tiktoken_or_an_empty_cache_warms_again(tokenizer_root, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", _fake_warm_up(calls))
    win_runtime.warm_tokenizer_cache(tokenizer_root)
    # A rebuilt root with another tiktoken: the cached entry may not validate.
    (tokenizer_root.site_packages / "tiktoken_ext" / "openai_public.py").write_text(
        "v2 with a different expected hash", encoding="utf-8"
    )
    win_runtime.warm_tokenizer_cache(tokenizer_root)
    # An emptied cache, with a stamp that still matches.
    for entry in tokenizer_root.tokenizer_cache.iterdir():
        entry.unlink()
    win_runtime.warm_tokenizer_cache(tokenizer_root)
    assert len(calls) == 3


@pytest.mark.skipif(sys.platform != "win32", reason="the AppContainer probe is Windows-only")
def test_capability_probe_only_validates_the_root(tmp_path, monkeypatch):
    # The probe runs on the event loop and at test import time; it used to
    # revoke dead sessions' ACEs and spawn a downloading tokenizer warm-up.
    from nooa.runtime.sandbox.guards import _probe_windows

    root = RuntimeRoot(tmp_path / "rt")
    (root.package).mkdir(parents=True)
    (root.package / "__init__.py").write_text("", encoding="utf-8")
    root.python.write_bytes(b"")

    def forbidden(*args, **kwargs):
        raise AssertionError("the capability probe must not have side effects")

    monkeypatch.setattr(win_runtime, "recover_stale_grants", forbidden)
    monkeypatch.setattr(win_runtime, "warm_tokenizer_cache", forbidden)
    caps = _probe_windows(str(root.root))
    assert caps.filesystem == "appcontainer", caps.detail
