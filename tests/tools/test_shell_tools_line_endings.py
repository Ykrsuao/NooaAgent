# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File edits write the same bytes on every platform.

``write_text`` in text mode turned every "\\n" into ``os.linesep``: on Windows
one ``replace()`` rewrote a whole LF file as CRLF. Edits keep the behaviour
Linux always had, including the normalization of a CRLF file to LF that
``test_shell_tools_stale_match`` pins. None of these tests start a shell.
"""

from __future__ import annotations

import pytest

from nooa.tools.shell_tools import ShellTools


@pytest.fixture
async def shell(tmp_path):
    tools = ShellTools(cwd=tmp_path)
    yield tools
    await tools.close()


async def test_replace_by_text_keeps_an_lf_file_lf(shell, tmp_path):
    target = tmp_path / "sample.py"
    target.write_bytes(b"alpha = 1\nbeta = 2\n")
    await shell.replace("sample.py", "alpha = 1", "alpha = 10")
    assert target.read_bytes() == b"alpha = 10\nbeta = 2\n"


async def test_replace_by_match_keeps_an_lf_file_lf(shell, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"one\ntwo\nthree\n")
    match = await shell.read("sample.txt", lines=(2, 2))
    await shell.replace(match, "TWO\nAND MORE\n")
    assert target.read_bytes() == b"one\nTWO\nAND MORE\nthree\n"


async def test_a_crlf_file_is_written_as_lf_on_every_platform(shell, tmp_path):
    # Universal-newline reads normalize it, as read() and Match anchors show.
    target = tmp_path / "sample.txt"
    target.write_bytes(b"one\r\ntwo\r\n")
    await shell.replace("sample.txt", "two", "2")
    assert target.read_bytes() == b"one\n2\n"


async def test_write_file_writes_line_endings_verbatim(shell, tmp_path):
    await shell.write_file("lf.txt", "x\ny\n")
    await shell.write_file("crlf.txt", "x\r\ny\r\n")
    assert (tmp_path / "lf.txt").read_bytes() == b"x\ny\n"
    assert (tmp_path / "crlf.txt").read_bytes() == b"x\r\ny\r\n"
