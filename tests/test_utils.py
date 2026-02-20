from pydantic_ai import ToolReturnPart
from pydantic_ai import ModelRequest

from pydantic_ai import TextPart, ModelResponse, ModelMessage, RunContext, ToolCallPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

from partial_mcp.toolset.utils import (
    extract_query_from_context,
    extract_message_history_from_context,
    UserMessage,
    AssistantMessage,
)
from partial_mcp.mcp_servers.retail.agent import get_agent


def const_agent(
    messages: list[ModelMessage],
    agent_info: AgentInfo,
) -> ModelResponse:
    if isinstance(messages[-1], ModelRequest):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content="response")])

    return ModelResponse(parts=[ToolCallPart(tool_name="tool")])


async def test_extract_query():
    agent = await get_agent([])
    tool_context = {}

    def tool(ctx: RunContext):
        tool_context["query"] = extract_query_from_context(ctx)

    with agent.override(
        model=FunctionModel(const_agent),
        tools=[tool],
    ):
        await agent.run(user_prompt="request")
    assert tool_context["query"] == "request"


async def test_extract_message_history():
    agent = await get_agent([])
    tool_context = {}

    def tool(ctx: RunContext):
        tool_context["history"] = extract_message_history_from_context(ctx)

    with agent.override(
        model=FunctionModel(const_agent),
        tools=[tool],
    ):
        history = (await agent.run(user_prompt="1")).all_messages()
        history = (
            await agent.run(user_prompt="2", message_history=history)
        ).all_messages()
        await agent.run(user_prompt="3", message_history=history)
    assert tool_context["history"] == [
        UserMessage(content="1"),
        AssistantMessage(content="response"),
        UserMessage(content="2"),
        AssistantMessage(content="response"),
        UserMessage(content="3"),
    ]
