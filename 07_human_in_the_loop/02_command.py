from langgraph.graph import StateGraph,START,END
from langgraph.types import Command
from typing import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State):
    print("Node A")
    return Command(
        goto="node_b",
        update={
            "text":state["text"]+"_a"
        }
    )

def node_b(state: State):
    print("Node B")
    return Command(
        goto="node_c",
        update={
            "text":state["text"]+"_b"
        }
    )

def node_c(state: State):
    print("Node C")
    return Command(
        goto=END,
        update={
            "text":state["text"]+"_c"
        }
    )


graph = StateGraph(State)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)

graph.set_entry_point('node_a')

app      = graph.compile()
app.get_graph().draw_mermaid_png(output_file_path='02_graph.png')

response = app.invoke({
    'text': "Maksim"
})

print(response)
