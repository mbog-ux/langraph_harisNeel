from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, add_messages, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(model='gpt-4o-mini')

# Define nodes

GENERATE_POST       = 'generate_post'
GET_REVIEW_DECISION = 'get_review_decision'
POST                = 'post'
COLLECT_FEEDBACK    = 'collect_feedback'


def generate_post(state: AgentState) -> AgentState:
    return {
        "messages": llm.invoke(state['messages'])
    }

def get_review_decision(state: AgentState) -> AgentState:
    post_content = state['messages'][-1].content

    print('\n Current post: ')
    print(post_content)
    print('\n')

    decision = input("Post to LinkedIn? (yes/no): ")
    if decision.lower() == 'yes':
        return 'post'
    elif decision.lower() == 'no':
        return  'feedback'

def collect_feedback(state: AgentState) -> AgentState:
    user_feedback = input('Provide your feedback: ')
    return {
        'message':[HumanMessage(content = user_feedback )]
    }

def post_node(state: AgentState) -> AgentState:
    print('Doing following post: \n')
    print(state['messages'][-1].content)
    print('\nPost has been aproved and posted')

    return state


graph = StateGraph(AgentState)

graph.add_node("generate_post", generate_post)
graph.add_node("collect_feedback", collect_feedback)
graph.add_node("get_review_decision", get_review_decision)  # now a real node
graph.add_node("post_node", post_node)


graph.add_edge(START, "generate_post")
graph.add_edge("generate_post", "get_review_decision")
graph.add_conditional_edges(
    "get_review_decision",
    get_review_decision,                 # router reads state produced by node
    {
        "post": "post_node",
        "feedback": "collect_feedback",
    },
)
graph.add_edge("collect_feedback", "generate_post")
graph.add_edge("post_node", END)

app   = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='01_graph.png')
