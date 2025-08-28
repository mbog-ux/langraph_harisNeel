from typing import TypedDict,Annotated,List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage
import json

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')
tool_search = TavilySearch(search_depth = 'basic')
tools = [tool_search]
llm  = llm.bind_tools(tools = tools)


def chatbot(state: AgentState)->AgentState:
    return {
        'messages':[llm.invoke(state['messages'])]
    }


def tools_router(state:AgentState)-> AgentState:
    if hasattr(state['messages'][-1],"tool_calls") and len(state['messages'][-1].tool_calls) > 0:
        return 'continue'
    else: 
        return 'end'
    

graph = StateGraph(AgentState)

graph.set_entry_point('chatbot')
graph.add_node('chatbot',chatbot)
graph.add_node('tool_node',ToolNode(tools= tools))

graph.add_conditional_edges(
    'chatbot',
    tools_router,
    {
        'continue':'tool_node',
        'end':END
    }
)

graph.add_edge('tool_node','chatbot')

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='02_graph.png')

while True:
    user_input = input("\nUser: ")
    if (user_input) in ['exit','quit','end']:
        break
    else:
        response = app.invoke({
            'messages':[HumanMessage(content = user_input)]
        })
        
        last_message = response['messages'][-1]

        if isinstance(last_message, tuple):
            print(last_message)
        else:
            print(f'AI: {last_message.content}')

        print(response)
        # with open("response.json", "w", encoding="utf-8") as f:
        #     f.write(json.dump(response))

# print(f'\nAI: {result.content}')
# print('\n')

# print(result.model_dump_json())

# with open("response.json", "w", encoding="utf-8") as f:
#     f.write(result.model_dump_json())
