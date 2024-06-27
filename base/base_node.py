from abc import ABC, abstractmethod
from typing import Final

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class BaseNode(ABC):

    # The chat roles:
    SYSTEM: Final[str] = "system"
    USER: Final[str] = "human"
    AI: Final[str] = "ai"
    PLACEHOLDER: Final[str] = "placeholder"

    # The history keys:
    HISTORY: Final[str] = "history"
    INPUT: Final[str] = "input"

    def __init__(self):
        self._model = self._make_model()
        self._prompt = self._make_prompt()
        self._output_parser = self._make_output_parser()
        self._chain = self._make_chain()

    @classmethod
    def _make_model(cls) -> ChatOllama:
        return ChatOllama(
            model="llama3",
            format="json",
            temperature=0,
            keep_alive=-1,
        )

    @classmethod
    def _make_prompt(cls) -> ChatPromptTemplate:

        return ChatPromptTemplate.from_messages(
            [
                (cls.SYSTEM, "You are an helpful AI assistant."),
                (cls.USER, "Can you help me write a story?"),
                (cls.AI, "Yes I can help you write a story. What is the story about?"),
                (cls.PLACEHOLDER, f"{{{cls.HISTORY}}}"),
                (cls.USER, f"{{{cls.INPUT}}}"),
            ]
        )

    @classmethod
    def _make_output_parser(cls) -> BaseOutputParser:
        return JsonOutputParser()

    @classmethod
    def _make_chain(cls) -> RunnableSerializable:
        return cls._make_prompt() | cls._make_model() | cls._make_output_parser()

    @abstractmethod
    def run(self):
        pass
