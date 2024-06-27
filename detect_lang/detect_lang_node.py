from enum import Enum

from structure.state import CreativeWriterState, MessageName
from detect_lang.detect_lang_model import chain
from detect_lang.detect_lang_prompts import LANGUAGE
from structure.prompts import INPUT


class EnglishOrNot(str, Enum):
    ENGLISH = "en"
    NOT_ENGLISH = "not_en"


def detect_language(state: CreativeWriterState) -> EnglishOrNot:
    """
    Detect the language of the text in the last message of the state.
    :param state: The state with the text to detect the language of.
    :return: Whether the language is English or not.
    """

    text_to_decide = state["steps"][-1]["message"].content

    if not text_to_decide:
        raise ValueError("No text to decide on the language")

    result = chain.invoke({INPUT: text_to_decide})

    detected = result.get(LANGUAGE)

    try:
        decision = EnglishOrNot(detected)
    except ValueError:
        raise ValueError(f"Invalid language detected: {detected}")

    if decision == EnglishOrNot.NOT_ENGLISH:
        state["steps"][-1]["message"].name = MessageName.NEED_TRANSLATION
    else:
        state["steps"][-1]["message"].name = MessageName.INPUT

    return decision


if __name__ == "__main__":

    print(EnglishOrNot(""))

    # print(chain.invoke({"input": "Das ist ein"}))
    # state = CreativeWriterState(messages=[])
    # state["messages"] = [{"content": "Hello, how are you?"}]
    # state = detect_language(state)
    # print(state["messages"][-1].content)  # "{'language': 'en'}
