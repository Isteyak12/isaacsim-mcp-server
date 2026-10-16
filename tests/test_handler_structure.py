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

"""Validate that the adapter and handler structure is correct."""

import ast
import os

EXTENSION_ROOT = os.path.join(os.path.dirname(__file__), "..", "isaac.sim.mcp_extension", "isaac_sim_mcp_extension")


def _parse_file(path):
    with open(path) as f:
        return ast.parse(f.read())


def test_adapter_base_has_all_abstract_methods():
    """Verify the base adapter defines all required abstract methods."""
    tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                    methods.add(node.name)
                elif isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                    methods.add(node.name)
    expected = {
        "get_stage",
        "get_assets_root_path",
        "discover_environments",
        "load_environment",
        "create_prim",
        "delete_prim",
        "add_reference_to_stage",
        "set_prim_transform",
        "get_prim_transform",
        "list_prims",
        "get_prim_info",
        "create_xform_prim",
        "create_articulation",
        "discover_robots",
        "get_robot_joint_info",
        "set_joint_positions",
        "get_joint_positions",
        "create_world",
        "create_simulation_context",
        "create_physics_scene",
        "create_camera",
        "capture_camera_image",
        "create_lidar",
        "get_lidar_point_cloud",
        "create_pbr_material",
        "create_physics_material",
        "apply_material",
        "create_light",
        "modify_light",
        "clone_prim",
        "import_urdf",
        "play",
        "pause",
        "stop",
        "step",
        "execute_script",
        # Observability methods (issue #1)
        "get_simulation_state",
        "get_physics_state",
        "get_joint_config",
        "reload_script",
        # Dimensional data (issue #2)
        "get_prim_actual_size",
    }
    assert methods == expected, f"Missing: {expected - methods}, Extra: {methods - expected}"


def test_v5_adapter_implements_all_methods():
    """Verify v5 adapter implements every abstract method from base."""
    base_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    v5_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "v5.py"))

    base_methods = set()
    for node in ast.walk(base_tree):
        if isinstance(node, ast.FunctionDef) and node.name != "__init__":
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Name) and decorator.id == "abstractmethod") or (
                    isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod"
                ):
                    base_methods.add(node.name)

    v5_methods = set()
    for node in ast.walk(v5_tree):
        if isinstance(node, ast.FunctionDef):
            v5_methods.add(node.name)

    missing = base_methods - v5_methods
    assert not missing, f"v5 adapter missing implementations: {missing}"


def test_all_handler_modules_have_register():
    """Verify every handler module exposes a register(registry, adapter) function."""
    handlers_dir = os.path.join(EXTENSION_ROOT, "handlers")
    handler_files = [
        "scene.py",
        "objects.py",
        "lighting.py",
        "robots.py",
        "sensors.py",
        "materials.py",
        "assets.py",
        "simulation.py",
    ]
    for filename in handler_files:
        filepath = os.path.join(handlers_dir, filename)
        assert os.path.exists(filepath), f"Handler file missing: {filename}"
        tree = _parse_file(filepath)
        func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert "register" in func_names, f"{filename} missing register() function"


def test_v6_implements_all_abstract_methods():
    """IsaacAdapterV6 must concretely implement every @abstractmethod on the base."""
    base_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "base.py"))
    abstract_methods = set()
    for node in ast.walk(base_tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                    abstract_methods.add(node.name)
                elif isinstance(decorator, ast.Attribute) and decorator.attr == "abstractmethod":
                    abstract_methods.add(node.name)

    v6_tree = _parse_file(os.path.join(EXTENSION_ROOT, "adapters", "v6.py"))
    v6_methods = {node.name for node in ast.walk(v6_tree) if isinstance(node, ast.FunctionDef)}

    missing = abstract_methods - v6_methods
    assert not missing, f"IsaacAdapterV6 is missing abstract methods: {sorted(missing)}"


# ── error envelope preserves structured context ──────────────────────────────


def test_execute_command_keeps_extra_fields_on_error():
    """A handler's structured error context must reach the client.

    The envelope forwarded only `message` on the error branch, so anything a
    handler added alongside it was silently dropped. Measured live: create_lidar
    refused a poisoned path and offered `suggested_prim_path`, and the client
    received None — the one field that lets an agent recover without inventing a
    prim path. Unit tests call handlers directly and cannot see this; only a
    round trip through the extension can.
    """
    import sys
    import types

    ext_mod = sys.modules.get("isaac_sim_mcp_extension.extension")
    if ext_mod is None:
        import isaac_sim_mcp_extension.extension as ext_mod  # noqa: F401

    ext = ext_mod.MCPExtension.__new__(ext_mod.MCPExtension)
    ext._registry = {
        "t.fail": lambda **p: {
            "status": "error",
            "message": "nope",
            "suggested_prim_path": "/World/L_2",
            "prim_path": "/World/L",
        }
    }
    ext._stage_pending = types.MethodType(lambda self: False, ext)

    out = ext._execute_command({"type": "t.fail", "params": {}})

    assert out["status"] == "error"
    assert out["message"] == "nope"
    assert out["suggested_prim_path"] == "/World/L_2", "structured error context was dropped"
    assert out["prim_path"] == "/World/L"


def test_execute_command_error_without_extras_is_unchanged():
    import types

    import isaac_sim_mcp_extension.extension as ext_mod

    ext = ext_mod.MCPExtension.__new__(ext_mod.MCPExtension)
    ext._registry = {"t.plain": lambda **p: {"status": "error", "message": "just this"}}
    ext._stage_pending = types.MethodType(lambda self: False, ext)

    out = ext._execute_command({"type": "t.plain", "params": {}})

    assert out == {"status": "error", "message": "just this"}


# ── the live integration suite must be opt-in (review process finding) ───────


def test_integration_tests_do_not_arm_themselves_from_a_live_socket():
    """`uv run pytest` must never touch a running Isaac Sim.

    The gate probed localhost:8766 at import, so whenever Kit happened to be up
    the command CLAUDE.md advertises as "no Isaac Sim needed" silently armed 43
    destructive tests — clear_scene, deletes, play/stop, and a camera creation
    that burns the 6.0 session's one undeletable-first-camera slot. This
    repository's own retraction log names exactly that as a past source of false
    bug reports, and documenting the hazard did not disarm it.

    Reachability may still be *required* on top, but an explicit opt-in has to
    come first.
    """
    import ast
    import os

    path = os.path.join(os.path.dirname(__file__), "test_integration.py")
    with open(path) as f:
        tree = ast.parse(f.read())

    # Names assigned anywhere in the module from an environment read.
    env_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ("environ" in ast.dump(node.value) or "getenv" in ast.dump(node.value)):
            env_names.update(t.id for t in node.targets if isinstance(t, ast.Name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "requires_isaac" for t in node.targets
        ):
            referenced = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
            assert referenced & env_names, (
                "requires_isaac is decided by probing the socket alone; it must also "
                f"require an explicit opt-in environment variable (env-derived names: {env_names or 'none'})"
            )
            return
    raise AssertionError("requires_isaac not found")
