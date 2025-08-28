from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import START,END,StateGraph
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages.base import message_to_dict
import json
import sqlite3

#https://langchain-ai.github.io/langgraph/concepts/persistence/

load_dotenv()
sqlite_conn = sqlite3.connect('checkpoint.sqlite',check_same_thread = False)
memory = SqliteSaver(sqlite_conn) 

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

app.get_graph().draw_mermaid_png(output_file_path='03_graph.png')


while True: 
    user_input = input("User: ")
    if(user_input in ["exit", "end"]):
        break
    else: 
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        }, config=config)

        print("AI: " + result["messages"][-1].content)

snapshot = app.get_state(config=config)
history = app.get_state_history(config=config)
print(result)

result["messages"] = [message_to_dict(m) for m in result["messages"]]
with open("03_responses.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
