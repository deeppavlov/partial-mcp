"""
Toolset
-------
This module is used for custom tool handling logic.
"""

from typing import Any

from pydantic_ai.toolsets import CombinedToolset, ToolsetTool
from pydantic_ai.tools import RunContext

from .utils import extract_message_history_from_context, extract_query_from_context


class Toolset(CombinedToolset):
    """
    Wrapper for an iterable of toolsets.
    Its `toolsets` argument will contain all toolsets given to the agent.
    Its methods can be redefined to allow for custom tool handling.
    """

    async def prepare(self):
        """
        This function will be called only once during benchmark:
        when setting up the agent.

        You can do some pre-processing here. For example:
        - send toolset data to some endpoint in order to prepare it
        """
        # This is an example implementation of the prepare function
        # Currently it simply prints the total length of all tool descriptions
        # You can change this function however you wish
        count = 0
        for toolset in self.toolsets:
            for tool in (
                await toolset.get_tools(ctx=None)  # pyrefly: ignore[bad-argument-type]
            ).values():
                count += len(tool.tool_def.description or "")

        print(
            f"Total character length of tool descriptions: {count}.\n"
            f"Rough token estimate (assuming 4 chars/token): {count / 4}."
        )

    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]:
        """
        Get tools available to agent.
        This function will be called every time agent is run.

        You can use this to filter or completely change tools available to agent at runtime:
        - You can call Tool Selector here and filter the items of `original_tools` accordingly.
        - You can return tools required by Code Mode instead of original tools.
            See `pydantic`'s code mode implementation for an example:
            https://github.com/pydantic/pydantic-ai/blob/3490542f44368d2c935d725b5bfc0f542890401b/pydantic_ai_slim/pydantic_ai/toolsets/code_mode/__init__.py#L240-L283

        :param ctx: Run context.
            You can get user's prompt and messages in the history from this:
            https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.prompt
        :return: A dictionary mapping tool names to tool definitions.
        """
        # example of using extraction utils:
        user_query = extract_query_from_context(ctx)
        history = extract_message_history_from_context(ctx)

        if not isinstance(user_query, str) or not isinstance(history, list):
            raise RuntimeError("Extraction error")

        # in order to remove extra tools you can either modify original tools or return another dictionary
        original_tools = await super().get_tools(ctx)
        return original_tools

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext,
        tool: ToolsetTool,
    ) -> Any:
        """
        Execute a tool.
        This function is called every time agent wants to execute a tool.

        Redefine this function if you're implementing code mode so that it runs tool discovery and code execution tools.

        :param name: Name of the tool agent wants to run (key of the `get_tools()` dictionary).
        :param tool_args: Argument of the tool call.
        :param ctx: Run Context.
        :param tool: Definition of the tool (value of the `get_tools()` dictionary).
        :return: Result of the tool call.
        """
        return await super().call_tool(name, tool_args, ctx, tool)

    def visit_and_replace(self, visitor):
        """
        Disable visit and replace method to prevent the creation of new objects
        of this class every agent run.

        This method is originally called every run to recreate DynamicToolset
        objects but ends up recreating CombinedToolset objects as well.
        """
        return self
