from typing import List

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph, StateGraph

from chains import revisor_chain, first_responder_chain
from execute_tools import execute_tools

def event_loop(state:List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item,ToolMessage) for item in state)

    num_iterations = count_tool_visits
    if num_iterations > 2:
        return 'end'
    else:
        return 'continue'

# graph = StateGraph(dict)
graph = MessageGraph()


graph.set_entry_point('draft')
graph.add_node('draft',first_responder_chain)
graph.add_node('execute_tools',execute_tools)
graph.add_node('revisor',revisor_chain)

graph.add_edge("draft","execute_tools")
graph.add_edge("execute_tools","revisor")

graph.add_conditional_edges(
    'revisor',
    event_loop,
    {
        'end':END,
        'continue':'execute_tools'
    })

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')    

response = app.invoke(
    "Write about how small buisnesse can laverage AI to grow"
)

print(response[-1].tool_calls[0]['args']['answer'])
# print(response)