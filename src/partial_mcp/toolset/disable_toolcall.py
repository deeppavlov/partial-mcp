"""
Disable Toolcall Toolset
------------------------
This module provides a toolset that disables tool execution of underlying toolset.
This is needed to ensure that irrelevant tools will not be called.
"""

from typing import Any

from pydantic_ai import RunContext, ToolsetTool

from pydantic_ai.toolsets import WrapperToolset


class RestrictedToolCallException(Exception):
    pass


class DisableToolcallToolset(WrapperToolset):
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        raise RestrictedToolCallException(f"Tried calling restricted tool {name}.")
