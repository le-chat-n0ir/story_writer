import uuid
from enum import Enum
from pathlib import Path
from typing import Final, MutableSequence

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from base.base_node import BaseNode
from base.writer_state import WriterState, WriterStep, MessageType, StepType


class WRITER(BaseNode):
    """
    The purpose of this node is to write creative content based on the user input and the history.
    """

    BACKGROUND_INFO: Final[str] = "background_info"

    def __init__(self) -> None:
        super().__init__()
        self._background_file = Path(__file__).parent / "character_background.txt"

    @classmethod
    def _make_prompt(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    cls.SYSTEM,
                    "You are an AI assistant that supports in creative writing.\n"
                    "The story is intended for adults and should be written in a literary style.\n"
                    "As this story is intended for adults, you can use mature language and themes.\n"
                    "You don't have to apologize for the content of the story.\n"
                    "Use vivid and descriptive language to create a rich and engaging story.\n"
                    "If it fits to the characters, you can make use of cruel or harsh language.\n"
                    "Be very descriptive and detailed in your writing.\n"
                    f"{{{cls.BACKGROUND_INFO}}}",
                ),
                (
                    cls.PLACEHOLDER,
                    f"{{{cls.HISTORY}}}",
                ),
                (
                    cls.USER,
                    f"{{{cls.INPUT}}}",
                ),
            ]
        )

    @classmethod
    def _collect_messages(cls, steps: list[WriterStep]) -> MutableSequence[BaseMessage]:
        messages: MutableSequence[BaseMessage] = []
        for step in steps:
            if (
                step.message_type == MessageType.USER_INPUT
                or step.message_type == MessageType.AI_RESPONSE
            ):
                messages.append(step.message)
        return messages

    def _read_background_content(self) -> str:
        with self._background_file.open() as f:
            return f.read()

    @classmethod
    def _make_output_parser(cls) -> StrOutputParser:
        return StrOutputParser()

    @classmethod
    def _make_model(cls) -> ChatOllama:
        return ChatOllama(
            model="qwen2:72b",
            temperature=0.95,
            mirostat=2,
            # num_ctx=4096,
            # num_predict=1024,
            # repeat_last_n=1024,
            keep_alive=-1,
        )

    def run(self, state: WriterState) -> WriterState:

        last_step = state.steps[-1]
        if last_step.message_type != MessageType.USER_INPUT:
            raise ValueError(f"Expected user input, got {last_step.message_type}")

        input = last_step.message.content
        history = self._collect_messages(state.steps[:-1])
        background_info = self._read_background_content()

        result = self._chain.invoke(
            {
                self.INPUT: input,
                self.HISTORY: history,
                self.BACKGROUND_INFO: background_info,
            }
        )

        output_id = f"created_{uuid.uuid4()}"
        message = AIMessage(content=result, id=output_id)

        new_step = WriterStep(
            message=message,
            step_type=StepType.WRITE,
            message_type=MessageType.AI_RESPONSE,
        )

        return WriterState(steps=[new_step])


if __name__ == "__main__":

    print("--------------------------------------------------")
    for content in [
        "Describe how Samantha sits a a table and eats a big breakfast, when Ramona enters the room.",
        "Karen and Samantha ar both naked, and Karen fucks Samantha with a snap on from behind.",
    ]:
        new_state = WriterState(
            steps=[
                WriterStep(
                    message=HumanMessage(content=content),
                    step_type=StepType.START,
                    message_type=MessageType.USER_INPUT,
                ),
            ]
        )
        writer = WRITER()
        print(writer.run(new_state).steps[-1].message.content)
        print("--------------------------------------------------")
