from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,END,START
load_dotenv()

class AgentState(TypedDict):
    count:int


def increment(state: AgentState) -> AgentState:
    return {
        "count":state["count"]+1
    }

def should_cotinue(state:AgentState) -> AgentState:
    if state['count'] < 5:
        return "continue"
    else:
        return 'end'
    
graph = StateGraph(AgentState)

graph.add_node('increment',increment)
graph.add_edge(START,'increment')
graph.add_conditional_edges(
    "increment",
    should_cotinue,
    {
        'continue':"increment",
        'end':END
    }
)

app = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='graph.png')

state = {
    "count" : 0
}

result = app.invoke(state) 
print(result)