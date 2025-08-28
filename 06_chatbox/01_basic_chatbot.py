from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

graph = StateGraph(AgentState)

graph.set_entry_point('chatbot')
graph.add_node('chatbot',chatbot)
graph.set_finish_point('chatbot')

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')

while True:
    user_input = input("\nUser: ")
    if (user_input) in ['exit','quit','end']:
        break
    else:
        response = app.invoke({
            'messages':[HumanMessage(content = user_input)]
        })
        
        last_messge = response['messages'][-1]

        if isinstance(last_messge, tuple):
            print(last_messge)
        else:
            print(f'AI: {last_messge.content}')

