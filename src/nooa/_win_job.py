# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job Object structures shared by the shell's process tree and the sandbox.

``nooa.tools._windows_job`` (owning a shell's descendants) and
``nooa.runtime.sandbox.win_api`` (confining a container worker) both set
``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``. A layout mistake there is silent,
since the kernel reads whatever bytes it is given, so the layout lives once.
Plain ctypes only: importable on every platform, used only on Windows.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class BasicLimit(ctypes.Structure):
    """``JOBOBJECT_BASIC_LIMIT_INFORMATION``."""

    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class IoCounters(ctypes.Structure):
    """``IO_COUNTERS``."""

    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class ExtendedLimit(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``."""

    _fields_ = [
        ("BasicLimitInformation", BasicLimit),
        ("IoInfo", IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]
