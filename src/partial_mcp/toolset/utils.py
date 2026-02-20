"""
Utils
-----
Utility functions to be used in `Toolset`.
"""

from pydantic_ai.messages import UserPromptPart
from dataclasses import dataclass

from pydantic_ai.tools import RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart


def extract_query_from_context(ctx: RunContext) -> str:
    """
    Extract user query from context.

    :param ctx: RunContext instance.
    :return: User query string.
    :raises RuntimeError: If user query string is empty or a sequence.
    """
    prompt = ctx.prompt
    if not isinstance(prompt, str):
        raise RuntimeError(f"Unexpected prompt type {type(prompt)}")
    return prompt


@dataclass
class UserMessage:
    """Message from user"""

    content: str
    kind: str = "user"


@dataclass
class AssistantMessage:
    """Reply from assistant"""

    content: str
    kind: str = "assistant"


def extract_message_history_from_context(
    ctx: RunContext,
) -> list[UserMessage | AssistantMessage]:
    """
    Extract message history from context.
    This includes the latest user query (the one from extract_query_from_context).
    This only contains text messages (user requests and agent responses).
    Tool calls, Images, system prompts, thoughts, e.t.c. are not included.

    :param ctx: RunContext instance.
    :return: Message history list.
    """
    history: list[AssistantMessage | UserMessage] = []
    for message in ctx.messages:
        content = "".join(
            part.content
            for part in message.parts
            if isinstance(part, (TextPart, UserPromptPart))
            if isinstance(part.content, str)
        )
        if content:
            if isinstance(message, ModelRequest):
                history.append(UserMessage(content=content))
            elif isinstance(message, ModelResponse):
                history.append(AssistantMessage(content=content))
            else:
                raise RuntimeError(f"Unexpected message type {type(message)}")

    return history
