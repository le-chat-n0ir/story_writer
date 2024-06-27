import uuid
from enum import Enum
from typing import TypedDict, Annotated, Union, Literal, Final

from langchain_core.load import Serializable
from langchain_core.load.serializable import (
    SerializedConstructor,
    SerializedNotImplemented,
)
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


# Message names:
class MessageName:
    NEED_TRANSLATION: Final[str] = "needs_translation"
    INPUT: Final[str] = "input"


class StateStep(TypedDict):
    step_type: Literal[
        "needs_translation",
        "input",
        "input_translate",
        "translate",
        "chatbot",
        "tool",
        "output",
    ]
    message: BaseMessage


def add_state_step(left: list[StateStep], right: list[StateStep]) -> list[StateStep]:
    # assign missing ids
    for step in left:
        if step["message"].id is None:
            step["message"].id = str(uuid.uuid4())
    for step in right:
        if step["message"].id is None:
            step["message"].id = str(uuid.uuid4())
    # merge
    left_idx_by_id = {step["message"].id: ii for ii, step in enumerate(left)}
    merged = left.copy()
    for step in right:
        if (existing_idx := left_idx_by_id.get(step["message"].id)) is not None:
            merged[existing_idx] = step
        else:
            merged.append(step)
    return merged


class CreativeWriterState(TypedDict):
    # messages: Annotated[list[BaseMessage], add_messages]
    steps: Annotated[list[StateStep], add_state_step]
