import operator
from enum import Enum
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel


class StepType(str, Enum):
    UNKNOWN = "unknown"
    START = "start"
    NODE = "node"
    TRANSLATE = "translate"
    LANGUAGE = "language"
    WRITE = "write"


class MessageType(str, Enum):
    UNKNOWN = "unknown"
    USER_INPUT = "input"
    AI_RESPONSE = "response"
    TRANSLATED = "translated"
    NEEDS_TRANSLATION = "needs_translation"


class WriterStep(BaseModel):
    message: AIMessage | HumanMessage | BaseMessage
    step_type: StepType = StepType.UNKNOWN
    message_type: MessageType = MessageType.UNKNOWN

    # @classmethod
    # def add(
    #     cls, left: list["WriterStep"], right: list["WriterStep"]
    # ) -> list["WriterStep"]:
    #     for elem in right:
    #         left.append(elem)
    #     return left


def add(a, b):
    # raise Exception("This function should not be called")
    print("---------------------------ADD---------------------------")
    return operator.add(a, b)


class WriterState(BaseModel):
    steps: Annotated[list[WriterStep], add]

    def messages(self) -> Sequence[BaseMessage]:
        return [step.message for step in self.steps]
