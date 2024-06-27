import uuid

from langchain_core.messages import HumanMessage

from structure.state import CreativeWriterState, StateStep
from structure.prompts import INPUT
from translate.translate_node import TRANSLATION, translate_chain


def translate(state: CreativeWriterState) -> CreativeWriterState:

    text_to_translate = state["steps"][-1]["message"].content

    result = translate_chain.invoke({INPUT: text_to_translate})

    translated_text = result.get(TRANSLATION)

    if not translated_text:
        raise ValueError(f"Translation failed: {result}")

    input_id = str(uuid.uuid4())
    message = HumanMessage(content=translated_text, id=input_id)

    return CreativeWriterState(
        steps=[
            StateStep(
                step_type="translate",
                message=message,
            )
        ],
    )
