from typing import TypedDict,List, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph,END,START
import operator
load_dotenv()

class AgentState(TypedDict):
    count:int
    sum: Annotated[int,operator.add]
    history: Annotated[List[int],operator.concat]
    #sum: int
     #history: List[int]


def increment(state: AgentState) -> AgentState:
    new_count = state['count']+1
    return {
        "count":new_count,
        "sum":new_count,
        'history': [new_count]
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
    "count" : 0,
    "sum" : 0,
    "history": [],
}

result = app.invoke(state) 
print(result)