from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, add_messages
from typing import TypedDict, Annotated

from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

load_dotenv()
search_tool = TavilySearch(depth='basic')
tools = [search_tool]

llm_base = ChatOpenAI(model='gpt-4o-mini')
llm_with_tools = llm_base.bind_tools(tools = tools)


class ChildState(TypedDict):
    messages : Annotated[list, add_messages]


def agent(state: ChildState) -> ChildState:
    return {
        "messages":llm_with_tools.invoke(state['messages'])
    }

def tool_router(state: ChildState):
    if hasattr(state['messages'][-1],'tool_calls') and len(state['messages'][-1].tool_calls) > 0:
        return 'continue'
    else:
        return 'end'
    

subgraph = StateGraph(ChildState)

subgraph.set_entry_point('agent')
subgraph.add_node('agent',agent)
subgraph.add_node('tools',ToolNode(tools=tools))
subgraph.add_conditional_edges(
    'agent',
    tool_router,
    {
        'continue':'tools',
        'end':END
    }
)
subgraph.add_edge('tools','agent')


search_app = subgraph.compile()

search_app.get_graph().draw_mermaid_png(output_file_path='01_graph_child.png')

# response = search_app.invoke({
#     "messages": [HumanMessage(content='What is the wheather in milan now')]
# })

# print(response['messages'][-1].content)

### Direct Embeddening
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]

parent_graph = StateGraph(ParentState)

parent_graph.add_node('search_app', search_app)
parent_graph.set_entry_point('search_app')
parent_graph.set_finish_point('search_app')

parent_graph = parent_graph.compile()

# response = parent_graph.invoke({
#     "messages": [HumanMessage(content='What is the wheather in milan now')]
# })

# print(response['messages'][-1].content)


### By transformation

class ParentState(TypedDict):
    query: str
    response: str

def search_agent(state: ParentState)->ParentState:

    subgraph_input = {
        "messages" : [HumanMessage(content=state['query'])]
    }
    response = search_app.invoke(subgraph_input)
    return {
        "response":response['messages'][-1].content
    }

parent_graph = StateGraph(ParentState)

parent_graph.add_node('search_agent', search_agent)
parent_graph.set_entry_point('search_agent')
parent_graph.set_finish_point('search_agent')
parent_graph = parent_graph.compile()

response = parent_graph.invoke({
    "query": 'What is the wheather in milan now',
    "response":""
})

print(response)