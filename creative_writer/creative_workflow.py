from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

from structure.state import CreativeWriterState
from language_recognition.node import detect_language


class CreativeWriter:
    creative_write_model_name = "llama3:70b"
    translater_model_name = creative_write_model_name
    check_refuse_model_name = creative_write_model_name

    def __init__(self) -> None:

        # The Ollama models:
        self.creative_write_model = ChatOllama(
            model=self.creative_write_model_name,
            temperature=0.95,
            mirostat=2,
            # num_ctx=4096,
            # num_predict=1024,
            # repeat_last_n=1024,
            keep_alive=-1,
        )
        self.translater_model = ChatOllama(
            model=self.translater_model_name,
            temperature=0.8,
            keep_alive=-1,
        )
        self.check_refuse_model = ChatOllama(
            model=self.check_refuse_model_name,
            format="json",
            temperature=0,
            keep_alive=-1,
        )

        # The config to store the response history:
        self.thread_id = "creative_writer"
        self.config: RunnableConfig = {"configurable": {"thread_id": self.thread_id}}
        self.db_connection = "chat_memory.db"
        self.memory = SqliteSaver.from_conn_string(self.db_connection)

        # The graph builder:
        self.graph_builder: StateGraph = StateGraph(CreativeWriterState)
        self.graph: CompiledGraph = self._compile_graph()

    def translate(self, state: CreativeWriterState) -> CreativeWriterState:
        return state

    def write_next_scene(self, state: CreativeWriterState) -> CreativeWriterState:
        return state

    def _compile_graph(self) -> CompiledGraph:
        self.graph_builder.add_node("translate", self.translate)
        self.graph_builder.add_node("write_next_scene", self.write_next_scene)
        self.graph_builder.add_node("check_language", detect_language)
