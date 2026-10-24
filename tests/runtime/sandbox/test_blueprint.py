# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a worker blueprint accepts before any worker starts."""

from __future__ import annotations

import pytest

from nooa.runtime.sandbox.blueprint import BlueprintError, WorkerBlueprint


def test_agent_defined_in_the_running_script_is_refused_with_a_fix():
    # The Windows container worker's own __main__ is win_worker, so a class
    # from the parent's script used to fail later as "cannot resolve".
    blueprint = WorkerBlueprint(agent_module="__main__", agent_qualname="MyAgent")
    with pytest.raises(BlueprintError, match="Move the class into a module"):
        blueprint.require_named_module()


def test_agent_in_an_importable_module_is_accepted():
    blueprint = WorkerBlueprint(agent_module="pkg.agents", agent_qualname="MyAgent")
    blueprint.require_named_module()
