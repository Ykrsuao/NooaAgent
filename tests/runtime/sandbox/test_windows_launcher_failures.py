"""Failure paths of the AppContainer launcher, with the Win32 calls faked.

The real-container tests in ``test_windows_appcontainer.py`` only reach these
branches when an OS call fails, which they cannot arrange. Each test here
records the order of teardown steps instead.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="AppContainer launcher is Windows-only")


class _Fakes:
    def __init__(self, monkeypatch, tmp_path, *, assign_fails=False, handshake_fails=False):
        from nooa.runtime.sandbox import win_api, win_launcher
        from nooa.runtime.sandbox.win_runtime import RuntimeRoot

        self.events: list[str] = []
        events = self.events

        class Journal:
            def record(self, path, sid):
                pass

            def clear(self, keep=False):
                events.append(f"journal.clear(keep={keep})")

        class FakeJob:
            def assign(self, handle):
                if assign_fails:
                    raise OSError("assign failed")
                events.append("job.assign")

            def terminate(self, code=1):
                events.append("job.terminate")

            def close(self):
                events.append("job.close")

        class FakeChild:
            pid = 1234
            pi = SimpleNamespace(hProcess=1, hThread=2)

            def resume(self):
                events.append("child.resume")

            def close(self):
                events.append("child.close")

        class K32:
            def TerminateProcess(self, handle, code):
                events.append("TerminateProcess")
                return 1

            def SetHandleInformation(self, handle, mask, flags):
                events.append(f"inherit={bool(flags)}")
                return 1

            def WaitForSingleObject(self, handle, ms):
                return 0

            def LocalFree(self, ptr):
                return None

        def handshake(*args):
            if handshake_fails:
                raise RuntimeError("handshake failed")

        def spawn(*args, **kwargs):
            events.append("spawn")
            return FakeChild()

        monkeypatch.setattr(win_api, "create_profile", lambda name: None)
        monkeypatch.setattr(win_launcher, "container_sid", lambda name: object())
        monkeypatch.setattr(win_launcher, "_sid_text", lambda sid: "S-1-15-2-1")
        monkeypatch.setattr(win_launcher, "sid_from_string", lambda text: object())
        monkeypatch.setattr(win_launcher.AclJournal, "open", lambda *a: Journal())
        monkeypatch.setattr(win_launcher, "grant", lambda path, sid, access: None)
        monkeypatch.setattr(win_launcher, "revoke", lambda path, sid: events.append("revoke"))
        monkeypatch.setattr(win_launcher, "Job", FakeJob)
        monkeypatch.setattr(win_launcher, "_spawn", spawn)
        monkeypatch.setattr(win_launcher, "_k32", K32())
        monkeypatch.setattr(win_launcher, "_handshake", handshake)
        monkeypatch.setattr(win_launcher, "_agent_import_roots", lambda init: [])
        monkeypatch.setattr(
            win_launcher, "_delete_profile_quiet", lambda name: events.append("delete_profile")
        )
        monkeypatch.setattr(RuntimeRoot, "check", lambda self, **k: None)

        root_dir = tmp_path / "rt"
        root_dir.mkdir()
        self.root = RuntimeRoot(root_dir)
        self.start = win_launcher.start_container_worker

    def run(self):
        from nooa.runtime.sandbox.config import ResolvedSpec

        return self.start({}, ResolvedSpec(filesystem=True), self.root, container_name="t")


def test_a_failed_assign_terminates_the_suspended_child(monkeypatch, tmp_path):
    """Outside the job, KILL_ON_JOB_CLOSE cannot end the child; it must be killed."""
    fakes = _Fakes(monkeypatch, tmp_path, assign_fails=True)
    with pytest.raises(OSError, match="assign failed"):
        fakes.run()
    events = fakes.events
    assert "child.resume" not in events
    assert events.index("TerminateProcess") < events.index("child.close")
    assert "job.close" in events
    assert "revoke" in events  # the grants are still undone


def test_release_kills_the_job_before_revoking(monkeypatch, tmp_path):
    """An open handle survives a revoke, so descendants must be dead first."""
    fakes = _Fakes(monkeypatch, tmp_path)
    handles = fakes.run()
    fakes.events.clear()
    handles.proc.release()
    handles.conn.close()
    events = fakes.events
    assert events.index("job.terminate") < events.index("revoke")


def test_the_broker_pipe_is_inheritable_only_around_the_spawn(monkeypatch, tmp_path):
    """Another thread's all-handles CreateProcess must not catch the pipe."""
    fakes = _Fakes(monkeypatch, tmp_path)
    handles = fakes.run()
    handles.proc.release()
    handles.conn.close()
    events = fakes.events
    start = events.index("inherit=True")
    assert events[start : start + 3] == ["inherit=True", "spawn", "inherit=False"]


def test_a_failed_handshake_cleans_up_exactly_once(monkeypatch, tmp_path):
    """release() already revoked; the outer handler must not repeat it."""
    fakes = _Fakes(monkeypatch, tmp_path, handshake_fails=True)
    with pytest.raises(RuntimeError, match="handshake failed"):
        fakes.run()
    events = fakes.events
    assert events.count("revoke") == 1
    assert events.count("journal.clear(keep=False)") == 1
    assert events.count("delete_profile") == 1
