from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages, AnyMessage
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from IPython.display import Image, display
from langgraph.graph import START, StateGraph, END
from dotenv import load_dotenv
import os 

load_dotenv()
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


class State(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]

llm=ChatGroq(model="qwen-2.5-32b")


def generate_title(state):
    sys_message_title = SystemMessage(content="You are a helpful assistant, which provides a best title for the blog as per the user input ")
    return {'messages':llm.invoke([sys_message_title]+state['messages']), "language": "English"}

def generate_blog(state):
    sys_message_blog = SystemMessage(content="You are a helpful assistant, write a detailed blog on the topic, strictly includes only content relevant to the title")
    return {'messages':llm.invoke([sys_message_blog]+state['messages']), "language": "English"}


def build_graph():
    graph_builder = StateGraph(State)

    graph_builder.add_node("Blog Title", generate_title)
    graph_builder.add_node("Blog Generator", generate_blog)

    graph_builder.add_edge(START,"Blog Title")
    graph_builder.add_edge("Blog Title","Blog Generator")
    graph_builder.add_edge("Blog Generator",END)

    graph=graph_builder.compile()
    return graph

agent = build_graph()