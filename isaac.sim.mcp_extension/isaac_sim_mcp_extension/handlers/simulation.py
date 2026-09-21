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

"""Simulation control command handlers."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from ..adapters.base import IsaacAdapterBase


def register(registry: Dict[str, Any], adapter: IsaacAdapterBase) -> None:
    registry["simulation.play"] = lambda **p: play(adapter, **p)
    registry["simulation.pause"] = lambda **p: pause(adapter, **p)
    registry["simulation.stop"] = lambda **p: stop(adapter, **p)
    registry["simulation.step"] = lambda **p: step(adapter, **p)
    registry["simulation.set_physics"] = lambda **p: set_physics(adapter, **p)
    registry["simulation.execute_script"] = lambda **p: execute_script(adapter, **p)
    registry["simulation.get_state"] = lambda **p: get_simulation_state(adapter, **p)
    registry["simulation.get_logs"] = lambda **p: get_logs(adapter, **p)
    registry["simulation.get_physics_state"] = lambda **p: get_physics_state_handler(adapter, **p)
    registry["simulation.get_joint_config"] = lambda **p: get_joint_config_handler(adapter, **p)
    registry["simulation.set_drive_params"] = lambda **p: set_drive_params_handler(adapter, **p)
    registry["simulation.reload_script"] = lambda **p: reload_script_handler(adapter, **p)


def play(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        adapter.play()
        return {"status": "success", "message": "Simulation started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def pause(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        adapter.pause()
        return {"status": "success", "message": "Simulation paused"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def stop(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        adapter.stop()
        return {"status": "success", "message": "Simulation stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def step(
    adapter: IsaacAdapterBase,
    num_steps: int = 1,
    observe_prims: Optional[Sequence[str]] = None,
    observe_joints: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    try:
        # num_steps reached three different code paths unvalidated and each did
        # something else with a negative count: V5 and Newton run
        # `for _ in range(num_steps)`, which is an empty loop, so the tool
        # reported "Stepped -5 frames" for a call that advanced nothing, while
        # V6/PhysX handed it to SimulationManager.step and errored. Neither is
        # an answer to "step backwards", which no engine supports.
        try:
            num_steps = int(num_steps)
        except (TypeError, ValueError):
            return {"status": "error", "message": f"num_steps must be an integer, got {num_steps!r}."}
        if num_steps < 1:
            return {
                "status": "error",
                "message": (
                    f"num_steps must be 1 or more, got {num_steps}. Physics cannot be stepped "
                    "backwards; call stop_simulation to return to the spawn state."
                ),
            }

        # Fail loud: stepping is only valid on a frozen (paused/stopped)
        # timeline. If a free run is active, N frames cannot be counted
        # exactly, so refuse rather than silently race the play loop.
        state = adapter.get_simulation_state()
        timeline_state = state.get("timeline_state") if isinstance(state, dict) else None
        if timeline_state == "playing":
            return {
                "status": "error",
                "message": (
                    "Cannot step while the simulation is running. A free-running "
                    "timeline is active — call pause_simulation or stop_simulation "
                    "first. Do not call play_simulation during the debug loop; "
                    "step_simulation is for a frozen timeline."
                ),
            }
        result = adapter.step(num_steps=num_steps, observe_prims=observe_prims, observe_joints=observe_joints)
        out = {
            "status": "success",
            "message": f"Stepped {num_steps} frames",
            "timeline_state": timeline_state,
            **result,
        }
        # The server instructions call this the debug loop, so it is the joint
        # read an agent leans on hardest — it has to carry the same provenance
        # get_joint_positions does. Only when joints were actually observed:
        # otherwise the tag is noise on every step call.
        if observe_joints:
            out = _tag_joint_source(adapter, out)
        return out
    except Exception as e:
        return {"status": "error", "message": str(e)}


def set_physics(
    adapter: IsaacAdapterBase,
    gravity: Optional[Sequence[float]] = None,
    time_step: Optional[float] = None,
    gpu_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    try:
        applied = []
        unsupported = []
        if gravity is not None:
            adapter.create_physics_scene(gravity=gravity)
            applied.append("gravity")
        # time_step and gpu_enabled are accepted by the signature but no adapter
        # implements them. They used to be swallowed silently under a blanket
        # "Physics parameters updated", so a caller could set a time step, be
        # told it worked, and run the whole session at the default rate. Say so
        # instead of pretending.
        if time_step is not None:
            unsupported.append("time_step")
        if gpu_enabled is not None:
            unsupported.append("gpu_enabled")

        if not applied and not unsupported:
            return {"status": "error", "message": "No physics parameters supplied"}
        if unsupported:
            # Status follows what actually happened to the stage. A partial
            # apply is a success that names what it dropped: reporting `error`
            # after writing gravity told the caller nothing had changed while
            # the scene was already on Mars. Only a request where *nothing*
            # could be applied is a failure.
            return {
                "status": "success" if applied else "error",
                "message": (
                    f"Applied: {applied or 'nothing'}. Not supported by this adapter and therefore "
                    f"ignored: {unsupported}. Set them directly with execute_script if you need them."
                ),
                "applied": applied,
                "unsupported": unsupported,
            }
        return {"status": "success", "message": f"Physics parameters updated: {applied}", "applied": applied}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def execute_script(adapter: IsaacAdapterBase, code: Optional[str] = None, cwd: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not code:
            return {"status": "error", "message": "code is required"}
        result = adapter.execute_script(code, cwd=cwd)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_simulation_state(adapter: IsaacAdapterBase) -> Dict[str, Any]:
    try:
        result = adapter.get_simulation_state()
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_physics_state_handler(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        result = adapter.get_physics_state(prim_path)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


_ECHO_WARNING = (
    "These joint values are authored drive targets, not simulated positions — the physics "
    "view could not serve the read, so they echo the last set_joint_positions call. A robot "
    "that looks perfectly converged here may not have moved at all. Step the simulation and "
    "retry, and check get_isaac_logs."
)


def _tag_joint_source(adapter: IsaacAdapterBase, result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the provenance of a joint read, and a warning when it is an echo.

    get_joint_positions has carried this since the read could fall back to
    drive targets; the two other paths that read joints did not. That matters
    most for get_joint_config, which reports
    position_error = target - actual: when the echo answers, actual *is* the
    target, so the error is identically 0.0 and reads as perfectly converged at
    the moment the read is least trustworthy.
    """
    try:
        source = adapter.joint_position_source
    except Exception:
        return result
    result["position_source"] = source
    if source != getattr(adapter, "JOINT_SOURCE_PHYSICS", "physics"):
        result["warning"] = _ECHO_WARNING
    return result


def get_joint_config_handler(adapter: IsaacAdapterBase, prim_path: Optional[str] = None) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        result = adapter.get_joint_config(prim_path)
        result = _tag_joint_source(adapter, result)
        if result.get("position_source") != getattr(adapter, "JOINT_SOURCE_PHYSICS", "physics"):
            # target - actual is identically zero when actual is the target.
            # Keeping it would report "converged" for a read that measured
            # nothing, which is the wrong answer rather than an imprecise one.
            for joint in result.get("joints", []) or []:
                joint.pop("position_error", None)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def engine_gain_write_warning(adapter: IsaacAdapterBase) -> Optional[str]:
    """Warn when a gain written to USD will not reach the running solver.

    Newton builds its model once and drive gains are not part of the
    stage-divergence check, so an authored gain lands on the stage and the
    running MuJoCo solver keeps the old one. Measured on 6.0.1 Newton, 0/5
    attempts changed `model.joint_target_ke`: it stayed at 3437746.75 through
    20 further steps while get_joint_config reported the new value, so the
    response described a sim that was not running. PhysX picks the write up on
    the next step (5/5 on both 6.0.1 and 5.1.0).

    The write is still worth performing — it persists on the stage and applies
    to the next model build — but a caller tuning a gain has to be told why
    nothing moved.
    """
    if adapter.engine != adapter.ENGINE_NEWTON:
        return None
    return (
        "Gain written to USD, but the running Newton solver will not pick it up — its model is built "
        "once and drive gains are not part of the stage-divergence check. The value persists on the "
        "stage and applies on the next model build; for gain tuning against a live sim, use the PhysX "
        "engine (isaac-sim.sh)."
    )


def set_drive_params_handler(
    adapter: IsaacAdapterBase,
    prim_path: Optional[str] = None,
    joint_name: Optional[str] = None,
    stiffness: Optional[float] = None,
    damping: Optional[float] = None,
    max_force: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        if not prim_path:
            return {"status": "error", "message": "prim_path is required"}
        if not joint_name:
            return {"status": "error", "message": "joint_name is required"}
        if stiffness is None and damping is None and max_force is None:
            return {
                "status": "error",
                "message": "nothing to set: pass at least one of stiffness, damping or max_force",
            }
        result = adapter.set_drive_params(
            prim_path,
            joint_name,
            stiffness=stiffness,
            damping=damping,
            max_force=max_force,
        )
        warnings = list(result.pop("warnings", []) or [])
        engine_warning = engine_gain_write_warning(adapter)
        if engine_warning:
            warnings.append(engine_warning)
        response: Dict[str, Any] = {"status": "success", **result}
        if warnings:
            response["warnings"] = warnings
        return response
    except Exception as e:
        return {"status": "error", "message": str(e)}


def reload_script_handler(
    adapter: IsaacAdapterBase, file_path: Optional[str] = None, module_name: Optional[str] = None
) -> Dict[str, Any]:
    try:
        if not file_path:
            return {"status": "error", "message": "file_path is required"}
        result = adapter.reload_script(file_path, module_name=module_name)
        return {"status": "success", **result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Log buffer for get_logs ───────────────────────────────────────────────────

# _log_buffer holds only [PRINT] output captured from execute_script /
# reload_script. WARN/ERROR come from Kit's own log file (see get_kit_log_path)
# — never from a Python log consumer, which deadlocks physics loads.
_log_buffer: list = []
_log_listener_active: bool = False
_play_boundary: int = 0
_MAX_LOG_BUFFER = 500
# Path of Kit's session log file: None = not resolved yet, "" = unavailable.
_kit_log_path: Optional[str] = None
# Byte offset into that file at the last timeline Play.
_kit_log_play_offset: int = 0


def append_log(entry: str) -> None:
    """Append an entry to the shared log buffer, trimming to the cap."""
    _log_buffer.append(entry)
    if len(_log_buffer) > _MAX_LOG_BUFFER:
        # Keep the boundary consistent when we drop from the front.
        global _play_boundary
        _log_buffer.pop(0)
        if _play_boundary > 0:
            _play_boundary -= 1


def mark_play_boundary() -> None:
    """Record the run boundary at the current timeline Play.

    Two positions: the [PRINT] buffer index, and the byte offset into Kit's log
    file, so `since_last_play` scopes both sources to the current run.
    """
    global _play_boundary, _kit_log_play_offset
    _play_boundary = len(_log_buffer)
    path = get_kit_log_path()
    if path:
        try:
            _kit_log_play_offset = os.path.getsize(path)
        except Exception:
            pass


def _select_logs(buffer: list, boundary: int, since_last_play: bool, count: int) -> list:
    """Pure selector: entries after the Play boundary (optional), capped to count."""
    scoped = buffer[boundary:] if since_last_play else buffer
    return scoped[-count:]


def get_kit_log_path() -> Optional[str]:
    """Absolute path of the log file Kit is writing this session, or None.

    Kit publishes it in the `/log/file` setting; fall back to the newest
    kit_*.log under the Omniverse logs tree.
    """
    global _kit_log_path
    if _kit_log_path is not None:
        return _kit_log_path or None
    path = None
    try:
        import carb

        value = carb.settings.get_settings().get("/log/file")
        if value and os.path.isfile(value):
            path = value
    except Exception:
        pass
    if path is None:
        try:
            candidates = glob.glob(os.path.expanduser("~/.nvidia-omniverse/logs/Kit/*/*/kit_*.log"))
            if candidates:
                path = max(candidates, key=os.path.getmtime)
        except Exception:
            path = None
    _kit_log_path = path or ""
    return path


def _ensure_log_listener():
    """Prepare log capture. Deliberately does NOT install a Python log consumer.

    A `carb`/`omni.log` message consumer is a Python callback that Kit invokes
    on whatever thread emitted the message. During a physics load
    (SingleArticulation.initialize() / World.initialize_physics()) omni.physx
    emits warnings from native TBB worker threads while the calling thread holds
    the GIL inside the native call — the worker blocks acquiring the GIL, the
    load never completes, and kit deadlocks permanently (reproduced on Isaac Sim
    5.1: spawning a Franka FR3, which emits invalid-inertia warnings, wedges kit
    forever with a Python consumer installed).

    Kit already writes every WARN/ERROR to its own log file starting at [0ms] —
    earlier than this extension can load, and it survives a crash or freeze — so
    get_logs reads that file instead. Startup-crash diagnostics are strictly
    better this way; nothing is captured on a live callback.
    """
    global _log_listener_active
    if _log_listener_active:
        return
    get_kit_log_path()
    _log_listener_active = True


def _read_kit_log_warnings(since_offset: int, count: int) -> Tuple[list, int]:
    """Return (WARN/ERROR lines from the kit log after `since_offset`, new offset)."""
    path = get_kit_log_path()
    if not path:
        return [], since_offset
    try:
        size = os.path.getsize(path)
        start = since_offset if 0 <= since_offset <= size else 0
        with open(path, "r", errors="replace") as f:
            f.seek(start)
            chunk = f.read()
            new_offset = f.tell()
    except Exception:
        return [], since_offset
    entries = [ln.rstrip("\n") for ln in chunk.splitlines() if ("[Warning]" in ln or "[Error]" in ln)]
    return entries[-count:], new_offset


def get_logs(
    adapter: IsaacAdapterBase, clear: bool = False, count: int = 100, since_last_play: bool = True
) -> Dict[str, Any]:
    """Return recent WARN/ERROR + [PRINT] log messages, scoped to the current run.

    WARN/ERROR are read from Kit's own session log file (covers everything from
    [0ms], survives a crash); [PRINT] comes from the captured stdout buffer.
    """
    try:
        global _kit_log_play_offset
        _ensure_log_listener()
        prints = _select_logs(_log_buffer, _play_boundary, since_last_play, count)
        warnings, _ = _read_kit_log_warnings(_kit_log_play_offset if since_last_play else 0, count)
        logs = (warnings + prints)[-count:]
        if clear:
            _log_buffer.clear()
            mark_play_boundary()
        return {
            "status": "success",
            "log_count": len(logs),
            "logs": logs,
            "kit_log_file": get_kit_log_path(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
