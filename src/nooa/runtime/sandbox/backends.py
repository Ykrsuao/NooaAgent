# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How a sandbox worker process is created, watched, and torn down.

The cell protocol is the same whichever mechanism confines the worker: send a
``run`` op, service brokered ``self.*`` calls, wait for a ``ResultDTO``, and
decide what a dead worker means. What differs is everything around it — how the
process comes into existence, how liveness is asked, what a kill returns, and
what the parent must undo afterwards.

:class:`CellBackend` is that seam. ``SandboxedExecutor`` holds one and never
asks which kind it is; :func:`open_backend` is the only place the decision is
made.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
from multiprocessing.connection import Connection
from typing import Any, Protocol

from nooa.runtime.sandbox.errors import (
    CellMemoryError,
    CellTimeoutError,
    SandboxUnavailable,
    WorkerDiedError,
)
from nooa.runtime.sandbox.guards import Capabilities
from nooa.runtime.sandbox.worker import worker_main

logger = logging.getLogger(__name__)


class CellBackend(Protocol):
    """One way to own a sandbox worker process.

    ``proc`` is opaque to the caller: each adapter supplies the object its own
    ``is_alive``/``exitcode``/``terminate`` understand, and the caller only ever
    passes it back. That keeps ``ContainerProcess`` and ``multiprocessing``
    processes interchangeable without either pretending to be the other.
    """

    def start(self, init: dict[str, Any]) -> tuple[Any, Connection]:
        """Create the worker and return ``(proc, conn)``.

        ``init`` is the first pipe message. It is always already picklable:
        the caller resolves the fork/spawn distinction before calling.
        """
        ...

    def is_alive(self, proc: Any) -> bool: ...

    def exitcode(self, proc: Any) -> int | None: ...

    def terminate(self, proc: Any) -> None:
        """Ask the worker to stop. Descendants may survive."""
        ...

    def kill(self, proc: Any) -> None:
        """Force the worker and everything it started to stop."""
        ...

    def join(self, proc: Any, timeout: float | None) -> None: ...

    def release(self, proc: Any) -> None:
        """Undo whatever the parent owns. A no-op for ordinary processes."""
        ...

    def classify_death(self, code: int | None) -> Exception | None:
        """Turn an exit code into a guardrail error, or ``None`` if unexplained."""
        ...

    def start_on_loop_thread(self) -> bool:
        """True when ``start`` must run on the event-loop thread.

        Forking from a worker thread duplicates a process whose other threads do
        not exist in the child, which is the documented way to deadlock a fork.
        Every other mechanism may start off-loop.
        """
        ...


class ProcessBackend:
    """``multiprocessing`` worker: fork or spawn, guarded from inside the child.

    The child calls ``install_guards`` itself, so the parent owns nothing but the
    process handle.
    """

    def __init__(self, start_method: str):
        self._start_method = start_method
        self._ctx = mp.get_context(start_method)

    def start(self, init: dict[str, Any]) -> tuple[Any, Connection]:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=worker_main,
            args=(child_conn, init),
            daemon=True,
            name="nooa-sandbox-worker",
        )
        proc.start()
        child_conn.close()
        return proc, parent_conn

    def is_alive(self, proc: Any) -> bool:
        return bool(proc.is_alive())

    def exitcode(self, proc: Any) -> int | None:
        return getattr(proc, "exitcode", None)

    def terminate(self, proc: Any) -> None:
        proc.terminate()

    def kill(self, proc: Any) -> None:
        proc.kill()

    def join(self, proc: Any, timeout: float | None) -> None:
        proc.join(timeout=timeout)

    def release(self, proc: Any) -> None:
        """Nothing to undo: the OS reclaims a joined process."""

    def classify_death(self, code: int | None) -> Exception | None:
        # Neither SIGXCPU nor SIGKILL exists on Windows; a missing constant must
        # not turn a worker death into an AttributeError here.
        sig_xcpu = getattr(signal, "SIGXCPU", None)
        sig_kill = getattr(signal, "SIGKILL", None)
        if sig_xcpu is not None and code == -sig_xcpu:
            return CellTimeoutError("cell exceeded its CPU-time limit and was killed")
        if sig_kill is not None and code == -sig_kill:
            return CellMemoryError(
                "worker was killed (out-of-memory or resource limit). "
                "Reduce the cell's memory use or raise max_memory_mb."
            )
        return None

    def start_on_loop_thread(self) -> bool:
        return self._start_method == "fork"


class ContainerBackend:
    """Windows AppContainer worker, created by the launcher rather than ``mp``.

    ``SECURITY_CAPABILITIES`` is a creation-time attribute, so ``multiprocessing``
    cannot enter the container. The launcher grants the ACLs, builds the Job
    Object, spawns suspended, assigns the job, resumes, and completes the ready
    handshake. What the parent gets back is a plain ``multiprocessing`` pipe plus
    a ``ContainerProcess`` that also knows how to revoke its own grants.
    """

    def __init__(self, config: Any, runtime_root: Any):
        self._config = config
        self._runtime_root = runtime_root
        self._summary = ""

    @property
    def runtime_summary(self) -> str:
        """What the last runtime-root preparation reported.

        Carried into startup failures: the worker's own error says what failed
        inside the sandbox, and this says whether the preconditions (package
        copy, tokenizer cache) were ever met.
        """
        return self._summary

    def start(self, init: dict[str, Any]) -> tuple[Any, Connection]:
        from nooa.runtime.sandbox.win_launcher import start_container_worker
        from nooa.runtime.sandbox.win_runtime import _workspace_packages, prepare_root

        # Logged, not discarded: the summary says whether the package copy and
        # the tokenizer cache are usable. Both are silent preconditions of the
        # launch, and a failure here shows up later as an opaque worker-start
        # error inside the sandbox.
        #
        # Every distribution this process imported goes into the root: a spawned
        # worker imports the agent's module by name, and an agent that lives in
        # a host package (nooa_cli's coding agent, for instance) would otherwise
        # fail with ModuleNotFoundError before its first cell.
        self._summary = prepare_root(self._runtime_root, sync=True, packages=_workspace_packages())
        logger.debug("sandbox runtime prepared: %s", self._summary)
        try:
            handles = start_container_worker(
                init,
                self._config.spec,
                self._runtime_root,
                # None lets the launcher mint one per start. The worker is
                # restarted after a kill, and a restarted worker is a new
                # process that should not inherit the dead one's identity.
                container_name=self._config.container_name,
                workspace=self._config.workspace,
                env_passthrough=self._config.env_passthrough,
            )
        except SandboxUnavailable as exc:
            raise SandboxUnavailable(f"{exc} [runtime: {self._summary}]") from exc
        return handles.proc, handles.conn

    def is_alive(self, proc: Any) -> bool:
        return bool(proc.is_alive())

    def exitcode(self, proc: Any) -> int | None:
        return getattr(proc, "exitcode", None)

    def terminate(self, proc: Any) -> None:
        proc.terminate()

    def kill(self, proc: Any) -> None:
        proc.kill()

    def join(self, proc: Any, timeout: float | None) -> None:
        proc.join(timeout)

    def release(self, proc: Any) -> None:
        """Revoke the ACLs and close the job/handles this launcher owns."""
        try:
            proc.release()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.debug("container worker release failed", exc_info=True)

    def classify_death(self, code: int | None) -> Exception | None:
        from nooa.runtime.sandbox.win_launcher import EXIT_JOB_TIME_LIMIT

        # A Job Object terminates the process with STATUS_QUOTA_EXCEEDED when the
        # per-job CPU limit is reached (measured by the M0 probe).
        if code == EXIT_JOB_TIME_LIMIT:
            return CellTimeoutError("cell exceeded its CPU-time limit and was killed")
        return None

    def start_on_loop_thread(self) -> bool:
        # The launcher creates the process itself (never forks), so starting it
        # off-loop is safe — and preferable: it waits for a ready handshake and
        # prepares the runtime root, which is seconds of blocking work.
        return False


def open_backend(
    config: Any,
    spec: Any,
    caps: Capabilities,
    *,
    runtime_root: Any = None,
) -> CellBackend:
    """Pick the backend this host and config can actually run.

    This is the only place the decision is made. Callers see a
    :class:`CellBackend` and never branch on platform again.
    """
    uses_container = os.name == "nt" and caps.filesystem == "appcontainer"
    if uses_container:
        if config.start_method != "spawn":
            raise SandboxUnavailable(
                "the AppContainer backend requires start_method='spawn': the launcher "
                "creates the process itself and cannot inherit live objects"
            )
        return ContainerBackend(_ContainerConfig(config, spec), runtime_root)
    return ProcessBackend(config.start_method)


class _ContainerConfig:
    """The slice of ``SandboxConfig`` the container launcher reads.

    Kept separate so the adapter does not depend on the whole config object:
    ``spec`` here is the pruned ``ResolvedSpec`` the executor settled on, which
    is not the same object the config carries.
    """

    def __init__(self, config: Any, spec: Any):
        self.spec = spec
        self.container_name = config.container_name
        self.workspace = config.workspace
        self.env_passthrough = config.env_passthrough


def require_start_method(config: Any) -> None:
    """Fail early when the configured start method does not exist here."""
    if config.start_method not in mp.get_all_start_methods():
        hint = (
            " On Windows use start_method='spawn': the container process is created by "
            "the AppContainer launcher, which cannot inherit live objects."
            if os.name == "nt"
            else " Run the agent in a supported Linux environment to use the sandbox backend."
        )
        raise SandboxUnavailable(
            f"Sandbox execution requires the {config.start_method!r} process start method, "
            f"which is unavailable on this platform.{hint}"
        )


def classify_unexplained(exc: WorkerDiedError, code: int | None, backend: CellBackend) -> Exception:
    """The adapter's verdict on an exit code, or the original error."""
    return backend.classify_death(code) or exc
