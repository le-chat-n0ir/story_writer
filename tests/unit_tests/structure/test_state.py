import unittest
import uuid

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

model = ChatOllama(
    model="llama3:70b",
    temperature=0.95,
    mirostat=2,
    keep_alive=-1,
)


def node(state: CreativeWriterState) -> CreativeWriterState:

    response = model.invoke([step["message"] for step in state["steps"]])
    if (response_id := response.id) is None:
        response_id = str(uuid.uuid4())
        response.id = response_id

    return CreativeWriterState(
        steps=[
            StateStep(
                step_type="chatbot",
                message=response,
            )
        ],
    )


config: RunnableConfig = {"configurable": {"thread_id": "1"}}
db_connection = "test_state.db"
memory = SqliteSaver.from_conn_string(db_connection)
graph_builder: StateGraph = StateGraph(CreativeWriterState)

graph_builder.add_node("node", node)
graph_builder.set_entry_point("node")
graph = graph_builder.compile(checkpointer=memory)


class TestCreativeWriterState(unittest.TestCase):

    def setUp(self) -> None:

        self.config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        self.db_connection = "test_state.db"
        self.memory = SqliteSaver.from_conn_string(self.db_connection)
        self.graph_builder: StateGraph = StateGraph(CreativeWriterState)

        self.graph_builder.add_node("node", node)
        self.graph_builder.set_entry_point("node")
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    # def test_init(self) -> None:
    #     state = CreativeWriterState(messages=[], step_types=[])
    #     self.assertEqual(state["messages"], [])
    #
    # def test_add_messages(self) -> None:
    #     state = CreativeWriterState(messages=[], step_types=[])
    #     message = BaseMessage(content="Hello", type="user")
    #     state["messages"].append(message)
    #     self.assertEqual(state["messages"], [message])
    #     isinstance(state["messages"][0], AIMessage)

    def test_with_graph(self):
        for user_input in [
            "Hi, my name is Lana. I'm 1.62m tall and weight 142 kg. "
            "I have a beautiful face (ask my girlfriend Mia!), long brown hair, "
            "green eyes, and a big smile. I'm a software engineer and I love to "
            "code; almost as much as I love to eat!",
            "What is my name?",
            "Describe me",
        ]:
            message = HumanMessage(content=user_input)
            events = self.graph.stream(
                CreativeWriterState(
                    steps=[
                        StateStep(
                            step_type="input",
                            step_id=message.id,
                            message=message,
                        )
                    ],
                ),
                self.config,
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

    # def test_read_memory(self) -> None:
    #     for entry in self.memory.list(None):
    #         entry: CheckpointTuple
    #         print(entry.config)
    #         print(entry.checkpoint["channel_values"])


if __name__ == "__main__":

    for user_input in [
        # "Hi, my name is Lana. I'm 1.62m tall and weight 142 kg. "
        # "I have a beautiful face (ask my girlfriend Mia!), long brown hair, "
        # "green eyes, and a big smile. I'm a software engineer and I love to "
        # "code; almost as much as I love to eat!",
        # "What is my name?",
        # "Describe me",
        "From time ti time I have to step onto the scale in front of Mia. She always has such a hungry look in her "
        "eyes, whenever I gained some weight."
        ""
    ]:
        input_id = str(uuid.uuid4())
        message = HumanMessage(content=user_input, id=input_id)
        events = graph.stream(
            CreativeWriterState(
                steps=[
                    StateStep(
                        step_type="input",
                        message=message,
                    )
                ],
            ),
            config,
            stream_mode="values",
        )
        for event in events:
            event["steps"][-1]["message"].pretty_print()
            # print(event["steps"])
            # print()
            # print()
