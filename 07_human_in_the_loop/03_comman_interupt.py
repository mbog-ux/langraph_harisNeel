from langgraph.graph import StateGraph,START,END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver 
from typing import TypedDict


memory = MemorySaver()

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

    human_response = interrupt("Do you want to go to C or D? Type C/D")

    if human_response == "C":
        return Command(
            goto="node_c",
            update={
                "text":state["text"]+"_b"
            }
        )
    elif human_response == 'D':
        return Command(
            goto="node_d",
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

def node_d(state: State):
    print("Node D")
    return Command(
        goto=END,
        update={
            "text":state["text"]+"_d"
        }
    )


graph = StateGraph(State)
config = {
    'configurable':{
        'thread_id':1
    }
}
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_node("node_c", node_c)
graph.add_node("node_d", node_d)


graph.set_entry_point('node_a')

app      = graph.compile(checkpointer = memory)
app.get_graph().draw_mermaid_png(output_file_path='03_graph.png')

# print(app.get_graph().draw_mermaid())


initialState = {'text': "Maksim"}
first_result = app.invoke(initialState, config = config, stream_mode="updates")
print(first_result)  
second_resuls = app.invoke(Command(resume='C'),config=config, stream_mode='upudates')