import operator
import unittest
import uuid
from typing import Sequence, Annotated

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.pydantic_v1 import BaseModel
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from base.writer_state import WriterState, WriterStep, StepType, MessageType

model = ChatOllama(
    model="llama3:70b",
    temperature=0.95,
    mirostat=2,
    keep_alive=-1,
)


def node(state: WriterState) -> WriterState:

    response = model.invoke(state.messages())

    return WriterState(
        steps=[
            WriterStep(
                message=response,
                step_type=StepType.NODE,
                message_type=MessageType.AI_RESPONSE,
            )
        ]
    )


config: RunnableConfig = {"configurable": {"thread_id": "1"}}
db_connection = "test_state.db"
memory = SqliteSaver.from_conn_string(db_connection)
graph_builder: StateGraph = StateGraph(WriterState)

graph_builder.add_node("node", node)
graph_builder.set_entry_point("node")
graph = graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":

    for entry in memory.list(config):
        print(entry)

    exit(0)

    for user_input in [
        "Hi, my name is Lana. I'm 1.62m tall and weight 142 kg. "
        "I have a beautiful face (ask my girlfriend Mia!), long brown hair, "
        "green eyes, and a big smile. I'm a software engineer and I love to "
        "code; almost as much as I love to eat!",
        "What is my name?",
        "Describe me",
    ]:
        message = HumanMessage(content=user_input)
        events = graph.stream(
            WriterState(
                steps=[
                    WriterStep(
                        message=message,
                        step_type=StepType.START,
                        message_type=MessageType.USER_INPUT,
                    ),
                ]
            ),
            config,
            stream_mode="values",
        )
        for event in events:
            event["steps"][-1].message.pretty_print()
