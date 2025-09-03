from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
    

workflow = StateGraph(ChildState)

workflow.set_entry_point('agent')
workflow.add_node('agent',agent)
workflow.add_node('tools',ToolNode(tools=tools))
workflow.add_conditional_edges(
    'agent',
    tool_router,
    {
        'continue':'tools',
        'end':END
    }
)
workflow.add_edge('tools','agent')


graph = workflow.compile()

graph.get_graph().draw_mermaid_png(output_file_path='01_graph.png')