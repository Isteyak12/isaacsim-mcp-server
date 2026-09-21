# MIT License
#
# Copyright (c) 2023-2025 omni-mcp
# Copyright (c) 2026 whats2000
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Writing a drive gain takes the unit the API reports, not USD's.

``get_joint_config`` reports an angular drive's gains per radian while USD
stores them per degree, and until this tool existed ``execute_script`` was the
only write path. Measured on the FR3, 5/5 attempts on both 6.0.1 PhysX and
5.1.0 PhysX: reading the reported ``3437746.77`` and authoring it straight back
moved the engine's own stiffness from ``3437746.75`` to ``196968384.00`` — a
57.30x inflation that raises no error.

Newton is the exception and it is measured, not assumed: there the USD write
lands but the running MuJoCo solver never picks it up (0/5 attempts, the
solver's ``joint_target_ke`` unchanged through 20 further steps), so the write
has to be reported with a warning rather than silently doing nothing to the
running sim.
"""

from __future__ import annotations

import ast
import math
import os
from unittest.mock import MagicMock

from isaac_sim_mcp_extension.handlers import simulation as sim

DEG_PER_RAD = 180.0 / math.pi


def test_handler_passes_the_callers_gain_to_the_adapter():
    adapter = MagicMock()
    adapter.engine = "physx"
    adapter.ENGINE_NEWTON = "newton"
    adapter.set_drive_params.return_value = {"joint": "fr3_joint1", "stiffness": 3437746.77}

    result = sim.set_drive_params_handler(
        adapter, prim_path="/World/Franka", joint_name="fr3_joint1", stiffness=3437746.77
    )

    assert result["status"] == "success"
    adapter.set_drive_params.assert_called_once()
    kwargs = adapter.set_drive_params.call_args.kwargs
    assert kwargs["stiffness"] == 3437746.77, "the adapter receives the per-radian value, unconverted"


def test_handler_requires_prim_path():
    adapter = MagicMock()
    adapter.engine = "physx"

    result = sim.set_drive_params_handler(adapter, joint_name="fr3_joint1", stiffness=1.0)

    assert result["status"] == "error"
    assert "prim_path" in result["message"]
    adapter.set_drive_params.assert_not_called()


def test_handler_requires_joint_name():
    adapter = MagicMock()
    adapter.engine = "physx"

    result = sim.set_drive_params_handler(adapter, prim_path="/World/Franka", stiffness=1.0)

    assert result["status"] == "error"
    assert "joint_name" in result["message"]
    adapter.set_drive_params.assert_not_called()


def test_handler_refuses_a_call_that_sets_nothing():
    """All three gains omitted is a no-op the caller almost certainly did not mean."""
    adapter = MagicMock()
    adapter.engine = "physx"

    result = sim.set_drive_params_handler(adapter, prim_path="/World/Franka", joint_name="fr3_joint1")

    assert result["status"] == "error"
    adapter.set_drive_params.assert_not_called()


def test_newton_write_is_reported_as_not_reaching_the_running_solver():
    """Measured 0/5 on 6.0.1 Newton: the USD write lands, the solver ignores it.

    `joint_target_ke` stayed at 3437746.75 through 20 further steps while
    get_joint_config reported the new value, so the response described a sim
    that was not running. Silence here would let a caller tune a gain on Newton
    and watch nothing change with nothing to explain it.
    """
    adapter = MagicMock()
    adapter.engine = "newton"
    adapter.ENGINE_NEWTON = "newton"
    adapter.set_drive_params.return_value = {"joint": "fr3_joint1", "stiffness": 3437746.77}

    result = sim.set_drive_params_handler(
        adapter, prim_path="/World/Franka", joint_name="fr3_joint1", stiffness=3437746.77
    )

    assert result["status"] == "success", "the USD write still happened and still persists"
    warnings = result.get("warnings") or []
    assert warnings, "Newton must warn that the running solver keeps its old gain"
    joined = " ".join(warnings).lower()
    assert "newton" in joined
    assert "solver" in joined or "running" in joined


def test_physx_write_carries_no_newton_warning():
    adapter = MagicMock()
    adapter.engine = "physx"
    adapter.ENGINE_NEWTON = "newton"
    adapter.set_drive_params.return_value = {"joint": "fr3_joint1", "stiffness": 3437746.77}

    result = sim.set_drive_params_handler(
        adapter, prim_path="/World/Franka", joint_name="fr3_joint1", stiffness=3437746.77
    )

    joined = " ".join(result.get("warnings") or []).lower()
    assert "newton" not in joined


# ── The command string and the adapter contract ─────────────────────────────


def test_command_is_registered_on_the_wire():
    registry: dict = {}
    sim.register(registry, MagicMock())

    assert "simulation.set_drive_params" in registry


def test_base_declares_set_drive_params():
    from isaac_sim_mcp_extension.adapters.base import IsaacAdapterBase

    assert "set_drive_params" in IsaacAdapterBase.__abstractmethods__, (
        "a version-specific write must be an abstract method, or one adapter can silently lack it"
    )


_ADAPTERS = os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension")


def _func_src(filename, name):
    with open(os.path.join(_ADAPTERS, "adapters", filename)) as f:
        src = f.read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in {filename}")


def test_both_adapters_implement_it():
    for adapter_file in ("v5.py", "v6.py"):
        assert _func_src(adapter_file, "set_drive_params"), adapter_file


def test_neither_adapter_writes_a_raw_gain():
    """The inverse of the read path: a per-radian gain must not reach USD unconverted.

    Authoring the reported value straight onto drive:angular:physics:stiffness
    is the 57.3x bug this tool exists to remove — measured 5/5 on 6.0.1 PhysX
    and 5/5 on 5.1.0, 3437746.75 -> 196968384.00.
    """
    for adapter_file in ("v5.py", "v6.py"):
        src = _func_src(adapter_file, "set_drive_params")
        assert "denormalize_gain" in src, f"{adapter_file} must convert before writing"
        assert "180" not in src, f"{adapter_file} must not open-code the ratio; units.py owns it"


def test_max_force_is_not_converted():
    """A torque on an angular drive, a force on a linear one — neither carries an angle."""
    for adapter_file in ("v5.py", "v6.py"):
        src = _func_src(adapter_file, "set_drive_params")
        for line in src.splitlines():
            if "max_force" in line and "denormalize_gain" in line:
                raise AssertionError(f"{adapter_file} converts max_force: {line.strip()}")


# ── The agent-facing tool ───────────────────────────────────────────────────


def _tool_src():
    path = os.path.join(os.path.dirname(__file__), "..", "isaac_mcp", "tools", "simulation.py")
    with open(path) as f:
        return f.read()


def test_tool_exists_and_sends_the_right_command():
    src = _tool_src()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_drive_params":
            body = ast.get_source_segment(src, node)
            assert "simulation.set_drive_params" in body, "tool must send the registered command string"
            args = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            assert {"prim_path", "joint_name", "stiffness", "damping", "max_force"} <= args
            return
    raise AssertionError("set_drive_params tool not found")


def test_tool_docstring_states_the_unit_it_accepts():
    """A writer that does not know the unit is the bug this tool removes."""
    src = _tool_src()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "set_drive_params":
            doc = ast.get_docstring(node) or ""
            assert "per-radian" in doc or "per radian" in doc
            assert "get_joint_config" in doc, "say the units match the tool that reports them"
            return
    raise AssertionError("set_drive_params tool not found")


def test_units_docstring_no_longer_promises_a_tool_that_does_not_exist():
    """units.py has claimed set_drive_params routes through the gain helpers
    since the read path landed. Now that it does, the claim has to be true."""
    path = os.path.join(_ADAPTERS, "adapters", "units.py")
    with open(path) as f:
        units_src = f.read()
    if "set_drive_params" in units_src:
        v5 = _func_src("v5.py", "set_drive_params")
        v6 = _func_src("v6.py", "set_drive_params")
        assert "denormalize_gain" in v5 and "denormalize_gain" in v6
