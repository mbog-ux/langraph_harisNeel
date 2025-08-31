from langgraph.graph import StateGraph, START,END, add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Annotated, TypedDict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

embedings = OpenAIEmbeddings()

class AgentState(TypedDict):
    messages: Annotated[List[str],add_messages]


llm = ChatOpenAI(model='gpt-4o-mini')

def topic_decision(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

def decide(state:AgentState) -> AgentState:
    if True:
        return 'on_topic'
    else:
        return 'off_topic'

def retrieve(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }
 
def generate_answer(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

def off_topic_reposne(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }


graph = StateGraph(AgentState)

graph.set_entry_point('topic_decision')
graph.add_node('topic_decision',topic_decision)
graph.add_node('retrieve',retrieve)
graph.add_node('generate_answer',generate_answer)
graph.add_node('off_topic_response',off_topic_reposne)


graph.add_conditional_edges(
    'topic_decision',
    topic_decision,
    {
        'on_topic':'retrieve',
        'off_topic':'off_topic_response',
    }
)

graph.add_edge('retrieve','generate_answer')
graph.add_edge('generate_answer',END)
graph.add_edge('off_topic_response',END)


app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='02_graph.png')