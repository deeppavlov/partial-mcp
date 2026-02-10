# Copyright Sierra
# This is a modified version of `tau2`'s `src/tau2/data_model/tasks.py`.
import textwrap
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class StructuredUserInstructions(BaseModel):
    """
    User instructions. This information defines the specific situation the user is in and the tasks they are trying to complete.
    """

    domain: Annotated[str, Field(description="The domain of the task.")]
    reason_for_call: Annotated[
        str, Field(description="The reason for the user to call the agent.")
    ]
    known_info: Annotated[
        Optional[str],
        Field(description="Known information about the user.", default=None),
    ]
    unknown_info: Annotated[
        Optional[str],
        Field(description="Unknown information about the user.", default=None),
    ]
    task_instructions: Annotated[str, Field(description="Instructions for the User.")]

    def __str__(self) -> str:
        lines = []
        tab = "\t"
        lines.append(f"Domain: {self.domain}")
        lines.append(f"Reason for call:\n{textwrap.indent(self.reason_for_call, tab)}")
        if self.known_info is not None:
            lines.append(f"Known info:\n{textwrap.indent(self.known_info, tab)}")
        if self.unknown_info is not None:
            lines.append(f"Unknown info:\n{textwrap.indent(self.unknown_info, tab)}")
        lines.append(
            f"Task instructions:\n{textwrap.indent(self.task_instructions, tab)}"
        )
        return "\n".join(lines)


UserInstructions = StructuredUserInstructions | str


class UserScenario(BaseModel):
    """
    User scenario. All the information that will be sent to the user simulator.
    """

    persona: None
    instructions: Annotated[
        UserInstructions,
        Field(
            description="Instructions for the User. This information defines the specific situation the user is in and the tasks they are trying to complete."
        ),
    ]


class Description(BaseModel):
    """
    Description of a scenario. This can be sent to the evaluator.
    """

    purpose: Annotated[
        Optional[str],
        Field(description="Explains what the scenario is testing.", default=None),
    ]
    relevant_policies: None
    notes: None


class Action(BaseModel):
    """
    An Agent/User action.
    Example:
      {
      "action_id": "get_user_details_1",
      "requestor": "assistant",
      "name": "get_user_details",
      "arguments": { "user_id": "sophia_silva_7557", "note": "I need to get the user details for user_id: sophia_silva_7557" },
      "compare_args": ["user_id"]
    },
    A tool call can be compared with an action by comparing the arguments in compare_args.
    If compare_args is None, will check all the arguments.
    """

    action_id: str = Field(
        description="The unique identifier for the action within a scenario."
    )
    name: str = Field(description="The name of the action.")
    arguments: dict = Field(description="The arguments for the action.")
    info: Optional[str] = Field(
        description="Information about the action.", default=None
    )


class EvaluationCriteria(BaseModel):
    """
    Evaluation criteria for a particular task. This will be sent to the evaluator.
    """

    actions: Annotated[
        Optional[list[Action]],
        Field(
            description="The actions that the agent should take to complete the task.",
            default=None,
        ),
    ]

    communicate_info: Annotated[
        Optional[list[str]],
        Field(
            description="List of information that the agent should communicate to the user.",
            default=None,
        ),
    ]

    nl_assertions: Annotated[
        Optional[list[str]],
        Field(
            description="List of assertions for the task, in natural language.",
            default=None,
        ),
    ]


class Task(BaseModel):
    """
    A task for a particular domain. This will be sent to the user simulator, the environment and the evaluator.
    """

    id: str = Field(description="The unique identifier for the task.")
    description: Annotated[
        Optional[Description],
        Field(
            description="Description of the task. This can be sent to the evaluator.",
            default=None,
        ),
    ]
    user_scenario: Annotated[
        UserScenario,
        Field(
            description="User scenario. This information will be sent to the user simulator."
        ),
    ]
    initial_state: None
    evaluation_criteria: Annotated[
        Optional[EvaluationCriteria],
        Field(
            description="Evaluation criteria for the task. This will be sent to the evaluator.",
            default=None,
        ),
    ]
