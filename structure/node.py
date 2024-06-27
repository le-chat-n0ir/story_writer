from abc import ABC, abstractmethod
from typing import Final

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable


class WriterNode(ABC):

    # The chat roles:
    SYSTEM: Final[str] = "system"
    USER: Final[str] = "human"
    AI: Final[str] = "ai"
    PLACEHOLDER: Final[str] = "placeholder"

    # The history keys:
    HISTORY: Final[str] = "history"
    INPUT: Final[str] = "input"

    json_model_name: str = "llama3"

    @classmethod
    def make_model(cls) -> ChatOllama:
        return ChatOllama(
            model=cls.json_model_name,
            format="json",
            temperature=0,
            keep_alive=-1,
        )

    @classmethod
    @abstractmethod
    def make_prompt(cls) -> ChatPromptTemplate:
        pass

    @classmethod
    @abstractmethod
    def make_chain(cls) -> RunnableSerializable:
        pass
