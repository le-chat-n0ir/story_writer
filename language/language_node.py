import uuid
from enum import Enum
from typing import Final

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from base.base_node import BaseNode
from base.writer_state import WriterState, WriterStep, MessageType, StepType


class EnglishOrNot(str, Enum):
    ENGLISH = "en"
    NOT_ENGLISH = "not_en"


class Language(BaseNode):
    """
    The purpose of this node is to detect whether the input language is English or not.
    """

    LANGUAGE: Final[str] = "language"

    @classmethod
    def _make_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    cls.SYSTEM,
                    "You are an AI assistant for language detection tasks.\n"
                    "Your only task is to detect if a text is written in english.\n"
                    f"Return the JSON with a single key '{cls.LANGUAGE}' with no preamble or explanation.\n"
                    f"If the text is in english, return '{cls.LANGUAGE}': 'en'.\n"
                    f"If the text is not in english, return '{cls.LANGUAGE}': 'not_en'.\n"
                    f"",
                ),
                (
                    cls.USER,
                    "Die Sonne scheint hell und warm. "
                    "Ein Schmetterling fliegt über eine Wiese, und setzt sich zärtlich auf eine Blume.",
                ),
                (
                    cls.AI,
                    "{{'language': 'not_en'}}",
                ),
                (
                    cls.USER,
                    "Lana set to the graceful wooden chair, and let her gaze wander through the café.",
                ),
                (
                    cls.AI,
                    "{{'language': 'en'}}",
                ),
                (
                    cls.USER,
                    f"{{{cls.INPUT}}}",
                ),
            ]
        )

    def run(self, state: WriterState) -> WriterState:
        raise NotImplementedError("This node is not meant to be run.")

    def condition(self, state: WriterState) -> EnglishOrNot:

        result = self._chain.invoke({self.INPUT: state.steps[-1].message.content})

        detected = result.get(self.LANGUAGE)

        try:
            decision = EnglishOrNot(detected)
        except ValueError:
            raise ValueError(f"Invalid language detected: {detected}")

        if decision == EnglishOrNot.NOT_ENGLISH:
            state.steps[-1].message_type = MessageType.NEEDS_TRANSLATION
        else:
            state.steps[-1].message_type = MessageType.USER_INPUT

        return decision


if __name__ == "__main__":

    print("--------------------------------------------------")
    for content in [
        "Die Sonne scheint hell und warm. Ein Schmetterling fliegt über eine Wiese, und setzt sich zärtlich auf eine "
        "Blume.",
        "Lana set to the graceful wooden chair, and let her gaze wander through the café.",
    ]:
        state = WriterState(
            steps=[
                WriterStep(
                    message=HumanMessage(content=content),
                    step_type=StepType.NODE,
                    message_type=MessageType.USER_INPUT,
                ),
            ]
        )
        language = Language()
        print(language.condition(state))
        print(state.steps[-1].message_type)
        print("--------------------------------------------------")
