from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import START,END,StateGraph
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
memory = MemorySaver() 

class AgentState(TypedDict):
    messages: Annotated[list,add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')

def chatbot(state: AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

graph = StateGraph(AgentState)

graph.set_entry_point('chatbot')
graph.add_node('chatbot',chatbot)
graph.set_finish_point('chatbot')

app = graph.compile(checkpointer=memory)

config = {
    "configurable":{
        "thread_id": 1
    }
}

response1 = app.invoke(
    {'messages':[HumanMessage(content = "Hello, I am Maksim")]
}, config = config)

response2 = app.invoke(
    {'messages':[HumanMessage(content = "What is my name?")]
},config = config)

app.get_graph().draw_mermaid_png(output_file_path='03_graph.png')
# print(results['messages'])
print(f'\nAI: ',response1['messages'][-1].content,'\n')
print(f'\nAI: ',response2['messages'][-1].content,'\n')

print(response2)
