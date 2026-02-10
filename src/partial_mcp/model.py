"""
Model
-----
This module defines an LLM model to be used by the agent.
"""

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


model = OpenAIChatModel(
    model_name=os.environ["MODEL_NAME"],
    provider=OpenAIProvider(
        api_key=os.environ["API_KEY"],
        base_url=os.environ["BASE_URL"],
    ),
)
