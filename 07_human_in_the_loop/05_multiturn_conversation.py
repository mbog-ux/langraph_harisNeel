from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.graph import StateGraph, END, add_messages, START
from langchain_openai import ChatOpenAI
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")

class State(TypedDict):
    linkedin_topic: str
    generated_post: Annotated[List[str],add_messages]
    human_feedback: Annotated[List[str],add_messages]


def model(state: State) -> State:
    print('[model] Generating content')
    linkedin_topic = state['linkedin_topic']
    feedback = state['human_feedback'] if 'human_feedback' in state else ["No feedback yer"] 

    prompt = f"""
        LinkedIn Topic: {linkedin_topic}
        Human Feedback: {feedback[-1] if feedback else 'No feedback yet'}

        Genereate a structures and well-writen LinkedIn post based on the given topic.

        Consider previous human feedback to refine the response
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn conente writer"),
        HumanMessage(content=prompt)
    ])

    generated_post = response.content

    print(f"[model_node] Generated post:\n {generated_post}")

    return {
        'generated_post':generated_post,
        'human_feedback':feedback
    }

def human_node(state: State):
    print('\n [human_node] awaiting human feedback...')

    generate_post = state['generated_post']

    user_feedback = interrupt({
        "generated_post":generate_post,
        "messages": "Provide feedack or type 'done' to finish"
    })

    print(f"[human_node] Revieved human feedback:\n {user_feedback}")

    if user_feedback == 'done':
        return Command(
            goto='end_node',
            update={"human_feedback":state["human_feedback"]+["Finilised"]}
        )
    return Command(
        goto   = 'model',
        update = {"human_feedback":state['human_feedback']+[user_feedback]}
    )

def end_node(state:State):
    """Final Node"""
    print(":\\n[end_node] Process Finished")
    print("Final Generated Post: ", state['generated_post'][-1])
    print("Final Human Feedback: ", state['human_feedback'])
    return {
        'generated_post':state['generated_post'],
        'human_feedback':state['human_feedback']
    }

graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("end_node", end_node)

graph.set_entry_point("model")

# Define the flow

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")

graph.set_finish_point("end_node")

# Enable Interrupt mechanism
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}

app.get_graph().draw_mermaid_png(output_file_path='05_graph.png')

linkedin_topic = input("Enter your LinkedIn topic: ")
initial_state = {
    "linkedin_topic": linkedin_topic, 
    "generated_post": [], 
    "human_feedback": []
}

for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        #  If we reach an interrupt, continuously ask for human feedback
        if(node_id == "__interrupt__"):
            while True: 
                user_feedback = input("Provide feedback (or type 'done' when finished): ")
                # Resume the graph execution with the user's feedback
                app.invoke(Command(resume=user_feedback), config=thread_config)
                # Exit loop if user says done
                if user_feedback.lower() == "done":
                    break