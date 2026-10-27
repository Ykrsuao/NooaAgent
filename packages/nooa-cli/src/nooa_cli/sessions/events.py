# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable metadata for interactive coding-agent sessions."""

from typing import Any, ClassVar

from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role


class SessionStarted(Metadata):
    """Identity and environment recorded once when a session is created."""

    _role: ClassVar[Role] = Role.METADATA

    origin: str = ""
    model: str = ""
    agent: str = ""
    working_directory: str = ""


class SessionTitleUpdated(Metadata):
    """The latest human- or agent-selected session title."""

    _role: ClassVar[Role] = Role.METADATA

    title: str = ""
    user_set: bool = False


class SessionUserMessage(Metadata):
    """Raw user text accepted by the agent runtime as a conversation turn."""

    _role: ClassVar[Role] = Role.METADATA

    content: str = ""
    # Set by hosts that record SessionTurnEnded. Without it, a missing end
    # marker says nothing: older sessions and other hosts never wrote one.
    tracks_end: bool = False


class SessionTurnEnded(Metadata):
    """The host finished handling the preceding user message, however it ended.

    Absent after a tracked message means the process stopped mid-turn.
    """

    _role: ClassVar[Role] = Role.METADATA


class SessionToolCard(Metadata):
    """A finished file-edit or terminal card, exactly as the live client got it.

    The activity events behind these cards are runtime-only and never stored,
    so the host keeps the rendered card for replay. Metadata never reaches the
    agent's context.
    """

    _role: ClassVar[Role] = Role.METADATA

    tool_call_id: str = ""
    card: dict[str, Any] = {}


SESSION_EVENT_TYPES: tuple[type[Metadata], ...] = (
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
    SessionTurnEnded,
    SessionToolCard,
)
