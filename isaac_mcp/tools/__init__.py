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

"""MCP tool modules for Isaac Sim."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from isaac_mcp.connection import IsaacConnection


class _CleanDocstrings:
    """Proxy for FastMCP whose tool() dedents each docstring before registering it.

    FastMCP sends the raw __doc__ as the tool description, and before Python
    3.13 that keeps the source indentation: ~3k characters of whitespace in
    every request, and a description size that depended on the interpreter.
    """

    def __init__(self, mcp: FastMCP) -> None:
        self._mcp = mcp

    def tool(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
        register = self._mcp.tool(*args, **kwargs)

        def decorator(fn: Callable[..., Any]) -> Any:
            if fn.__doc__:
                fn.__doc__ = inspect.cleandoc(fn.__doc__)
            return register(fn)

        return decorator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mcp, name)


def register_all_tools(mcp: FastMCP, get_connection: Callable[[], IsaacConnection]) -> None:
    """Register all MCP tools from submodules.

    Args:
        mcp: FastMCP server instance.
        get_connection: Callable that returns an IsaacConnection.
    """
    from . import assets, graphs, lighting, materials, objects, robots, scene, sensors, simulation

    for module in [scene, objects, lighting, robots, sensors, materials, assets, simulation, graphs]:
        module.register_tools(_CleanDocstrings(mcp), get_connection)
