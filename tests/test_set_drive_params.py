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


_ADAPTERS = os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension")


def _func_src(filename, name):
    with open(os.path.join(_ADAPTERS, "adapters", filename)) as f:
        src = f.read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


def test_one_shared_implementation_in_base():
    """Pure USD, so one copy: two drift apart and 5.1 and 6.0 write differently."""
    from isaac_sim_mcp_extension.adapters.base import IsaacAdapterBase

    assert "set_drive_params" not in IsaacAdapterBase.__abstractmethods__
    assert _func_src("base.py", "set_drive_params")
    for adapter_file in ("v5.py", "v6.py"):
        assert _func_src(adapter_file, "set_drive_params") is None, f"{adapter_file} must not shadow base"


def test_read_and_write_resolve_the_drive_through_one_helper():
    """If get_joint_config probed a different drive instance than the write,
    the write would land in a different unit from the one the read reported."""
    assert "find_joint_drive" in _func_src("base.py", "set_drive_params")
    for adapter_file in ("v5.py", "v6.py"):
        src = _func_src(adapter_file, "get_joint_config")
        assert "find_joint_drive" in src, adapter_file
        assert "DriveAPI.Get" not in src, f"{adapter_file} probes inline again"


def test_write_converts_and_does_not_open_code_the_ratio():
    """Authoring the reported value raw is the 57.3x bug this tool removes —
    measured 5/5 on 6.0.1 PhysX and 5/5 on 5.1.0, 3437746.75 -> 196968384.00."""
    src = _func_src("base.py", "set_drive_params")
    assert "denormalize_gain" in src
    assert "180" not in src, "units.py owns the ratio"


# ── Behaviour against a fake stage ──────────────────────────────────────────


class _Attr:
    def __init__(self, value=None, override=None):
        self.value, self.override = value, override

    def Set(self, v):
        self.value = v

    def Get(self):
        return self.override if self.override is not None else self.value


class _Drive:
    def __init__(self, override=None):
        self.attrs = {k: _Attr(override=override) for k in ("stiffness", "damping", "max_force")}

    def CreateStiffnessAttr(self):
        return self.attrs["stiffness"]

    def CreateDampingAttr(self):
        return self.attrs["damping"]

    def CreateMaxForceAttr(self):
        return self.attrs["max_force"]


class _Revolute:
    pass


class _Prismatic:
    pass


class _Prim:
    def __init__(self, path, kind=None, drives=None, children=()):
        self.path, self.kind, self.drives, self.children = path, kind, drives or {}, list(children)

    def IsValid(self):
        return True

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self):
        return self.path

    def IsA(self, cls):
        return self.kind is cls

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


class _Stage:
    def __init__(self, *roots):
        self.prims = {p.path: p for r in roots for p in r.walk()}

    def GetPrimAtPath(self, path):
        return self.prims.get(path) or MagicMock(IsValid=lambda: False)


def _install_fake_pxr(monkeypatch):
    import types

    usd = types.SimpleNamespace(PrimRange=lambda root: root.walk())
    drive_api = types.SimpleNamespace(Get=lambda prim, t: prim.drives.get(t))
    physics = types.SimpleNamespace(RevoluteJoint=_Revolute, PrismaticJoint=_Prismatic, DriveAPI=drive_api)
    monkeypatch.setitem(__import__("sys").modules, "pxr", types.SimpleNamespace(Usd=usd, UsdPhysics=physics))


def _robot(path, override=None):
    j1 = _Prim(f"{path}/fr3_joint1", _Revolute, {"angular": _Drive(override)})
    finger = _Prim(f"{path}/fr3_finger_joint1", _Prismatic, {"linear": _Drive(override)})
    bare = _Prim(f"{path}/no_drive", _Revolute)
    return _Prim(path, children=[j1, finger, bare])


def _write(monkeypatch, stage, **kw):
    from isaac_sim_mcp_extension.adapters.base import IsaacAdapterBase

    _install_fake_pxr(monkeypatch)
    fake_self = MagicMock(get_stage=lambda: stage)
    return IsaacAdapterBase.set_drive_params(fake_self, **kw)


def test_angular_gain_is_stored_per_degree_and_read_back(monkeypatch):
    robot = _robot("/World/A")
    out = _write(monkeypatch, _Stage(robot), prim_path="/World/A", joint_name="fr3_joint1", stiffness=1000.0)
    stored = robot.children[0].drives["angular"].attrs["stiffness"].value
    assert math.isclose(stored, 1000.0 / DEG_PER_RAD)
    assert math.isclose(out["stiffness"], 1000.0), "the response is the read-back, normalized"
    assert out["joint_path"] == "/World/A/fr3_joint1"
    assert "warnings" not in out


def test_linear_gain_and_max_force_pass_through(monkeypatch):
    robot = _robot("/World/A")
    out = _write(
        monkeypatch,
        _Stage(robot),
        prim_path="/World/A",
        joint_name="fr3_finger_joint1",
        stiffness=777.0,
        max_force=99.0,
    )
    attrs = robot.children[1].drives["linear"].attrs
    assert attrs["stiffness"].value == 777.0 and attrs["max_force"].value == 99.0
    assert out["drive_type"] == "linear"


def test_ambiguous_joint_name_is_refused(monkeypatch):
    """prim_path above two robots: writing the first match would silently miss the other."""
    world = _Prim("/World", children=[_robot("/World/A"), _robot("/World/B")])
    try:
        _write(monkeypatch, _Stage(world), prim_path="/World", joint_name="fr3_joint1", stiffness=1.0)
    except ValueError as e:
        assert "/World/A/fr3_joint1" in str(e) and "/World/B/fr3_joint1" in str(e)
    else:
        raise AssertionError("an ambiguous joint name must not write")


def test_stronger_opinion_is_reported_not_echoed(monkeypatch):
    """A weaker edit target leaves the old value in force; echoing the input would hide it."""
    robot = _robot("/World/A", override=5.0)
    out = _write(monkeypatch, _Stage(robot), prim_path="/World/A", joint_name="fr3_joint1", stiffness=1000.0)
    assert not math.isclose(out["stiffness"], 1000.0)
    assert out["warnings"] and "stronger layer" in out["warnings"][0]


def test_joint_without_a_drive_is_refused(monkeypatch):
    try:
        _write(monkeypatch, _Stage(_robot("/World/A")), prim_path="/World/A", joint_name="no_drive", stiffness=1.0)
    except ValueError as e:
        assert "no UsdPhysicsDriveAPI" in str(e)
    else:
        raise AssertionError("a joint with no drive must not report success")


def test_bad_values_are_refused_before_anything_is_written(monkeypatch):
    for kw in (
        {"stiffness": -400.0},
        {"damping": float("nan")},
        {"stiffness": float("inf")},
        {"stiffness": "abc"},
        {"max_force": -1.0},
    ):
        robot = _robot("/World/A")
        try:
            _write(monkeypatch, _Stage(robot), prim_path="/World/A", joint_name="fr3_joint1", **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} must be refused")
        assert all(a.value is None for a in robot.children[0].drives["angular"].attrs.values()), kw


def test_infinite_max_force_is_allowed(monkeypatch):
    """inf is USD's default maxForce and means unlimited."""
    robot = _robot("/World/A")
    out = _write(monkeypatch, _Stage(robot), prim_path="/World/A", joint_name="fr3_joint1", max_force=float("inf"))
    assert math.isinf(out["max_force"])


def test_max_force_is_not_converted_on_an_angular_drive(monkeypatch):
    """A torque carries no angle: 99 N·m stays 99, unlike the gains beside it."""
    robot = _robot("/World/A")
    _write(monkeypatch, _Stage(robot), prim_path="/World/A", joint_name="fr3_joint1", max_force=99.0)
    assert robot.children[0].drives["angular"].attrs["max_force"].value == 99.0


def test_applied_to_solver_is_a_field_not_only_a_warning():
    """An agent that skips warnings must still see that Newton kept the old gain."""
    for engine, expected in (("physx", True), ("newton", False)):
        adapter = MagicMock()
        adapter.engine = engine
        adapter.ENGINE_NEWTON = "newton"
        adapter.set_drive_params.return_value = {"joint": "fr3_joint1", "stiffness": 1000.0}
        result = sim.set_drive_params_handler(adapter, prim_path="/World/A", joint_name="fr3_joint1", stiffness=1000.0)
        assert result["applied_to_solver"] is expected, engine


def test_stage_override_is_not_applied_to_solver():
    adapter = MagicMock()
    adapter.engine = "physx"
    adapter.ENGINE_NEWTON = "newton"
    adapter.set_drive_params.return_value = {"stiffness": 5.0, "warnings": ["stiffness: a stronger layer opinion"]}
    result = sim.set_drive_params_handler(adapter, prim_path="/World/A", joint_name="fr3_joint1", stiffness=1000.0)
    assert result["applied_to_solver"] is False
    assert result["warnings"] == ["stiffness: a stronger layer opinion"]


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
    """units.py names set_drive_params as its write half; the claim has to be true."""
    path = os.path.join(_ADAPTERS, "adapters", "units.py")
    with open(path) as f:
        units_src = f.read()
    if "set_drive_params" in units_src:
        assert "denormalize_gain" in _func_src("base.py", "set_drive_params")
