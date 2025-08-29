from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,add_messages, START, END
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.prebuilt.tool_node import ToolNode
from langchain_tavily import TavilySearch


load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools  = [TavilySearch(depth='basic')]
llm    = ChatOpenAI(model='gpt-4o-mini').bind_tools(tools=tools)
memory = MemorySaver()

def model(state: AgentState) -> AgentState:
    return {
        "messages": llm.invoke(state['messages'])
    }

tool_node = ToolNode(tools=tools)

def should_continue(state:AgentState):
    if hasattr(state['messages'][-1],'tool_calls') and len(state['messages'][-1].tool_calls) >0:
        return 'continue'
    else:
        return 'end'


graph = StateGraph(AgentState)

graph.set_entry_point('model')
graph.add_node('model',model)
graph.add_node('tools',tool_node)

graph.add_conditional_edges(
    'model',
    should_continue,
    {
        'continue':'tools',
        'end':END
    }
)

graph.add_edge('tools','model')

config = {'configurable':{'thread_id':1}}
app   = graph.compile(checkpointer=memory,interrupt_before=['tools'])

events = app.stream({
    "messages":[HumanMessage("What is the current weather in Chennai?")]
},config = config, stream_mode='values')

for event in events:
    event["messages"][-1].pretty_print()

snapshot = app.get_state(config=config)

events = app.stream(None,config = config, stream_mode='values')

for event in events:
    event["messages"][-1].pretty_print()
# app.get_graph().draw_mermaid_png(output_file_path='04_graph.png')



