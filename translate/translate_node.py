import uuid
from typing import Final

from langchain.smith.evaluation.string_run_evaluator import StringExampleMapper
from langchain_core.load import Serializable
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable

from base.base_node import BaseNode
from structure.node import WriterNode
from structure.state import CreativeWriterState, StateStep
from base.writer_state import WriterState, WriterStep, MessageType, StepType


class Translate(BaseNode):
    TRANSLATION: Final[str] = "translation"

    def __init__(self):
        super().__init__()

    @classmethod
    def _make_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    cls.SYSTEM,
                    "You are an AI assistant for translation tasks.\n"
                    "Your only task is to translate a text into english.\n"
                    "Transform units of measurement, currency, and other numbers as needed.\n"
                    f"Return the JSON with a single key '{cls.TRANSLATION}' with no preamble or explanation.\n"
                    "Always return the translated text without any additional information.\n",
                ),
                (
                    cls.USER,
                    "Die Sonne scheint hell und warm. "
                    "Ein Schmetterling fliegt über eine Wiese, und setzt sich zärtlich auf eine Blume.",
                ),
                (
                    cls.AI,
                    "The sun is shining bright and warm. "
                    "A butterfly flies over a meadow, and gently settles on a flower.",
                ),
                (
                    cls.USER,
                    "Lana setzte sich auf den grazilen Holzstuhl, und ließ ihren Blick durch das Café schweifen.",
                ),
                (
                    cls.AI,
                    "Lana sat down on the graceful wooden chair, and let her gaze wander through the café.",
                ),
                (
                    cls.USER,
                    f"{{{cls.INPUT}}}",
                ),
            ]
        )

    def run(self, state: WriterState) -> WriterState:

        last_step = state.steps[-1]
        if last_step.message_type != MessageType.USER_INPUT:
            raise ValueError(f"Expected user input, got {last_step.message_type}")

        text_to_translate = last_step.message.content

        result = self._chain.invoke({self.INPUT: text_to_translate})

        translated_text = result.get(self.TRANSLATION)

        if not translated_text:
            raise ValueError(f"Translation failed: {result}")

        input_id = f"translated_{uuid.uuid4()}"
        message = HumanMessage(content=translated_text, id=input_id)

        new_step = WriterStep(
            message=message,
            step_type=StepType.TRANSLATE,
            message_type=MessageType.TRANSLATED,
        )

        return WriterState(steps=[new_step])


if __name__ == "__main__":

    translate = Translate()
    print(
        translate.run(
            WriterState(
                steps=[
                    WriterStep(
                        message=HumanMessage(
                            content="Die Sonne scheint hell und warm. "
                            "Ein Schmetterling fliegt über eine Wiese, und setzt sich zärtlich auf eine Blume."
                        ),
                        step_type=StepType.NODE,
                        message_type=MessageType.USER_INPUT,
                    ),
                ]
            )
        )
    )
    #
    # print(
    #     Translate.make_chain().invoke(
    #         {Translate.INPUT: "Die Sonne scheint hell und warm."}
    #     )
    # )
