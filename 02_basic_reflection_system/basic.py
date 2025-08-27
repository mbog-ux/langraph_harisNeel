from typing import TypedDict,List,Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import START, END, MessageGraph, StateGraph
from chains import reflection_chain,generation_chain

load_dotenv()

class AgentState():
    messages: List[BaseMessage]


graph = StateGraph(AgentState)
graph = MessageGraph()

def generate_node(state):
    response =  generation_chain.invoke({
        "messages":state
    })
    print(response)
    return response

def reflect_node(state):
    response = reflection_chain.invoke({
        "messages":state
    })
    return [HumanMessage(content=response.content)]

def should_continue(state):
    if (len(state)>3):
        return 'end'
    else:
        return 'continue'
    
def print_stream(stream):
    for s in stream:        
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


graph.set_entry_point('generate')
graph.add_node('generate',generate_node)
graph.add_node('reflect',reflect_node)

graph.add_conditional_edges(
    'generate',
    should_continue,
    {
        'end':END,
        'continue': 'reflect'
    }
)

graph.add_edge('reflect','generate')

app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='graph.png')
app.get_graph().print_ascii()

# inputs = HumanMessage(content="AI agents taking over content creation")
# print_stream(app.stream(inputs,stream_mode='values'))

response = app.invoke(HumanMessage(content="AI agents taking over content creation"))
print(response)