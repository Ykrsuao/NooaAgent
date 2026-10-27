# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared durable coding-agent session storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime

import nooa_cli.sessions.store as store_module
import pytest
from nooa_cli.sessions import InvalidSessionIdError, SessionNotFoundError, SessionStore

from nooa.interactive import AgentMessage
from nooa.storage import SessionAlreadyActiveError


def test_create_record_title_list_and_resume(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(
        session_id="session-one",
        origin="tui",
        model="test/model",
        agent="CodingAgent",
        working_directory="/workspace",
    )
    session.record_user_message("hello")
    session.events.add(AgentMessage(content="hi back"))
    session.set_title("First session", user_set=True)
    session.close()

    info = store.list()[0]
    assert info.id == "session-one"
    assert info.origin == "tui"
    assert info.model == "test/model"
    assert info.agent == "CodingAgent"
    assert info.working_directory == "/workspace"
    assert info.title == "First session"
    assert info.title_is_user_set is True
    assert info.turn_count == 1

    resumed = store.open("session-one")
    try:
        assert resumed.info == info
        assert [(turn.role, turn.content) for turn in resumed.turns()] == [
            ("user", "hello"),
            ("agent", "hi back"),
        ]
    finally:
        resumed.close()


def test_session_summary_uses_targeted_queries(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = store.create(session_id="query-shape")
    session.record_user_message("hello")
    session.set_title("Targeted")
    session.close()

    queries: list[str] = []
    real_connect = sqlite3.connect

    class RecordingConnection:
        def __init__(self, *args, **kwargs):
            self._connection = real_connect(*args, **kwargs)

        def execute(self, query, parameters=()):
            queries.append(" ".join(query.split()))
            return self._connection.execute(query, parameters)

        def close(self):
            self._connection.close()

    monkeypatch.setattr(store_module.sqlite3, "connect", RecordingConnection)

    assert store.get("query-shape").title == "Targeted"
    assert not any(
        query == "SELECT event_type, data FROM events ORDER BY insertion_order" for query in queries
    )
    assert any("SELECT COUNT(*)" in query for query in queries)
    assert sum("LIMIT 1" in query for query in queries) == 2


def test_different_session_databases_can_be_open_together(tmp_path):
    store = SessionStore(tmp_path)
    first = store.create(session_id="first")
    second = store.create(session_id="second")
    try:
        first.record_user_message("one")
        second.record_user_message("two")
        assert [turn.content for turn in first.turns()] == ["one"]
        assert [turn.content for turn in second.turns()] == ["two"]
    finally:
        first.close()
        second.close()


def test_same_live_session_cannot_be_opened_twice(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="live")
    try:
        with pytest.raises(SessionAlreadyActiveError):
            store.open("live")
    finally:
        session.close()

    resumed = store.open("live")
    resumed.close()


def test_user_messages_are_thread_safe_when_host_opts_into_cross_thread_access(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="threaded", check_same_thread=False)
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def write(prefix: str) -> None:
        try:
            barrier.wait(timeout=5)
            for index in range(25):
                session.record_user_message(f"{prefix}-{index}")
        except Exception as error:
            errors.append(error)

    writers = [threading.Thread(target=write, args=(prefix,)) for prefix in ("a", "b")]
    for writer in writers:
        writer.start()
    barrier.wait(timeout=5)
    for writer in writers:
        writer.join(timeout=10)

    try:
        assert errors == []
        assert all(not writer.is_alive() for writer in writers)
        assert len(session.turns()) == 50
        assert session.info.turn_count == 50
    finally:
        session.close()


def test_open_missing_or_invalid_session(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(SessionNotFoundError):
        store.open("missing")
    for unsafe in ("", ".", "..", "../escape", "nested/id", "nested\\id", "bad\x00id"):
        with pytest.raises(InvalidSessionIdError):
            store.path_for(unsafe)


def test_delete_refuses_live_session_then_removes_database_and_sidecars(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="delete-me")
    path = session.path
    with pytest.raises(SessionAlreadyActiveError):
        store.delete(session.id)

    session.close()
    path.with_name(f"{path.name}-wal").touch()
    path.with_name(f"{path.name}-shm").touch()
    assert store.delete("delete-me") is True
    assert not path.exists()
    assert not path.with_name(f"{path.name}-wal").exists()
    assert not path.with_name(f"{path.name}-shm").exists()
    assert store.delete("delete-me") is False


def test_delete_missing_session_from_missing_root_returns_false(tmp_path):
    store = SessionStore(tmp_path / "not-created")
    assert store.delete("missing") is False


def test_list_skips_corrupt_and_non_session_databases(tmp_path):
    store = SessionStore(tmp_path)
    valid = store.create(session_id="valid")
    valid.close()
    (tmp_path / "corrupt.db").write_bytes(b"not sqlite")
    connection = sqlite3.connect(tmp_path / "no-start.db")
    connection.execute("CREATE TABLE events (event_type TEXT, data TEXT, insertion_order INTEGER)")
    connection.close()

    assert [info.id for info in store.list()] == ["valid"]


def test_prefix_search_is_literal_and_sorted(tmp_path):
    store = SessionStore(tmp_path)
    for session_id in ("prefix-a", "prefix-b", "other"):
        session = store.create(session_id=session_id)
        session.close()

    assert set(store.find_by_prefix("prefix-")) == {"prefix-a", "prefix-b"}
    assert store.find_by_prefix("*") == []
    assert store.find_by_prefix("../") == []


def test_reads_legacy_tui_session_events(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="legacy-placeholder")
    session.close()
    store.delete("legacy-placeholder")

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE events (
            tag TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            data TEXT NOT NULL,
            insertion_order INTEGER NOT NULL
        );
        """
    )
    timestamp = datetime.now(UTC).isoformat()
    legacy_events = [
        (
            "TUISessionStart",
            {
                "event_type": "TUISessionStart",
                "timestamp": timestamp,
                "model": "legacy/model",
                "agent_cls": "TUIAgent",
                "working_dir": "/legacy",
            },
        ),
        (
            "TUIUserInput",
            {"event_type": "TUIUserInput", "timestamp": timestamp, "text": "old user"},
        ),
        (
            "TUIAgentMessage",
            {
                "event_type": "TUIAgentMessage",
                "timestamp": timestamp,
                "content": "old agent",
            },
        ),
        (
            "TUISessionRename",
            {
                "event_type": "TUISessionRename",
                "timestamp": timestamp,
                "name": "Legacy title",
                "user_named": True,
            },
        ),
    ]
    for order, (event_type, data) in enumerate(legacy_events):
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, 'active', ?, ?)",
            (str(order + 1), f"id-{order}", event_type, json.dumps(data), order),
        )
    connection.commit()
    connection.close()

    info = store.get("legacy")
    assert info.origin == "tui"
    assert info.model == "legacy/model"
    assert info.agent == "TUIAgent"
    assert info.working_directory == "/legacy"
    assert info.title == "Legacy title"
    assert info.title_is_user_set is True
    assert info.turn_count == 1
    assert [(turn.role, turn.content) for turn in store.load_turns("legacy")] == [
        ("user", "old user"),
        ("agent", "old agent"),
    ]


def _cell_events(session, call_id: str, code: str, **metadata) -> None:
    from nooa.context_blocks.events import ToolCallEvent

    session.events.add(
        ToolCallEvent(
            tool_call_id=call_id,
            name="execute_python",
            arguments={"code": code},
            metadata=metadata,
        )
    )


def _output_event(session, call_id: str, status: str, **fields) -> None:
    from nooa.context_blocks.events import ResultStatus
    from nooa.events import PythonOutput

    session.events.add(
        PythonOutput(
            tool_call_id=call_id,
            execution_status=ResultStatus(status),
            execution_count=1,
            **fields,
        )
    )


def test_interrupted_turn_returns_its_cells_in_order(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="interrupted")
    session.record_user_message("do three things", tracks_end=True)
    _cell_events(session, "prefill", "inspect()", prefill=True)
    _output_event(session, "prefill", "complete", stdout="inputs")
    _cell_events(session, "one", "write('a')")
    _output_event(session, "one", "complete", stdout="Created a\n")
    _cell_events(session, "two", "boom()")
    _output_event(session, "two", "error", error="NameError: boom")
    _cell_events(session, "prose", "# reply", synthetic=True)
    _cell_events(session, "three", "sleep(10)")
    session.close()

    cells = store.load_interrupted_cells("interrupted")
    assert [(c.tool_call_id, c.code, c.output, c.status) for c in cells] == [
        ("one", "write('a')", "Created a", "completed"),
        ("two", "boom()", "NameError: boom", "failed"),
        ("three", "sleep(10)", None, "interrupted"),
    ]


def _shape(item) -> str:
    from nooa_cli.sessions import SessionCard, SessionCell

    if isinstance(item, SessionCell):
        return f"cell:{item.tool_call_id}"
    if isinstance(item, SessionCard):
        return f"card:{item.tool_call_id}"
    return f"{item.role}:{item.content}"


def test_replay_keeps_cells_between_the_messages_they_ran_among(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="interleaved")
    session.record_user_message("look then fix", tracks_end=True)
    _cell_events(session, "look", "ls()")
    _output_event(session, "look", "complete", stdout="a.py\n")
    session.events.add(AgentMessage(content="Found it, fixing."))
    _cell_events(session, "fix", "sleep(10)")
    session.close()

    replay = store.load_replay("interleaved")
    assert [_shape(item) for item in replay.items] == [
        "user:look then fix",
        "cell:look",
        "agent:Found it, fixing.",
        "cell:fix",
    ]
    assert replay.interrupted_from == 0


def test_a_cancelled_cell_keeps_its_partial_output(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="cancelled-cell")
    session.record_user_message("go", tracks_end=True)
    _cell_events(session, "one", "loop()")
    _output_event(session, "one", "cancelled", stdout="partial\n")
    session.close()

    [cell] = store.load_interrupted_cells("cancelled-cell")
    assert (cell.output, cell.status) == ("partial", "interrupted")


def test_an_ended_turn_is_not_interrupted(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="ended")
    session.record_user_message("go", tracks_end=True)
    _cell_events(session, "one", "sleep(10)")
    session.record_turn_end()
    session.close()

    assert store.load_interrupted_cells("ended") is None


def test_a_killed_turn_with_no_cells_is_still_interrupted(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="no-cells")
    session.record_user_message("first", tracks_end=True)
    session.record_turn_end()
    session.record_user_message("second", tracks_end=True)
    session.close()

    assert store.load_interrupted_cells("no-cells") == []


def test_untracked_messages_are_never_judged_interrupted(tmp_path):
    """Sessions written before the marker existed have no end markers at all."""
    store = SessionStore(tmp_path)
    session = store.create(session_id="legacy")
    session.record_user_message("go")
    _cell_events(session, "one", "sleep(10)")
    session.close()

    assert store.load_interrupted_cells("legacy") is None


def test_only_the_last_turn_is_considered(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="later-turn")
    session.record_user_message("killed", tracks_end=True)
    _cell_events(session, "old", "sleep(10)")
    session.record_user_message("continue", tracks_end=True)
    session.record_turn_end()
    session.close()

    assert store.load_interrupted_cells("later-turn") is None


def test_replay_leaves_out_what_precedes_the_first_user_message(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="grouped")
    _cell_events(session, "orphan", "setup()")
    session.record_user_message("first")
    _cell_events(session, "a", "a()")
    session.record_user_message("second")
    _cell_events(session, "b", "b()")
    _output_event(session, "b", "complete", stdout="ok")
    session.close()

    replay = store.load_replay("grouped")
    assert [_shape(item) for item in replay.items] == [
        "user:first",
        "cell:a",
        "user:second",
        "cell:b",
    ]
    assert replay.interrupted_from is None


def test_replay_keeps_cards_in_place_and_the_latest_version_of_each(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="cards")
    session.record_user_message("go")
    _cell_events(session, "cell", "edit()")
    session.record_tool_card("edit-1", {"toolCallId": "edit-1", "title": "Edited a.py"})
    session.record_tool_card("cmd-1", {"toolCallId": "cmd-1", "title": "Running command"})
    _output_event(session, "cell", "complete")
    session.events.add(AgentMessage(content="done"))
    session.record_tool_card("cmd-1", {"toolCallId": "cmd-1", "title": "Ran command"})
    session.close()

    items = store.load_replay("cards").items
    assert [_shape(item) for item in items] == [
        "user:go",
        "cell:cell",
        "card:edit-1",
        "card:cmd-1",
        "agent:done",
    ]
    # The finished card replaces the running one, at the running one's place.
    assert items[3].card["title"] == "Ran command"
