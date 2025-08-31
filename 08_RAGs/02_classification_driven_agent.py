from langgraph.graph import StateGraph, START,END, add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Annotated, TypedDict
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

load_dotenv()

embedding_function = OpenAIEmbeddings()

docs = [
    Document(
        page_content="Peak Performance Gym was founded in 2015 by former Olympic athlete Marcus Chen. With over 15 years of experience in professional athletics, Marcus established the gym to provide personalized fitness solutions for people of all levels. The gym spans 10,000 square feet and features state-of-the-art equipment.",
        metadata={"source": "about.txt"}
    ),
    Document(
        page_content="Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM. On weekends, our hours are 7:00 AM to 9:00 PM. We remain closed on major national holidays. Members with Premium access can enter using their key cards 24/7, including holidays.",
        metadata={"source": "hours.txt"}
    ),
    Document(
        page_content="Our membership plans include: Basic (₹1,500/month) with access to gym floor and basic equipment; Standard (₹2,500/month) adds group classes and locker facilities; Premium (₹4,000/month) includes 24/7 access, personal training sessions, and spa facilities. We offer student and senior citizen discounts of 15% on all plans. Corporate partnerships are available for companies with 10+ employees joining.",
        metadata={"source": "membership.txt"}
    ),
    Document(
        page_content="Group fitness classes at Peak Performance Gym include Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates. Beginner classes are held every Monday and Wednesday at 6:00 PM. Intermediate and advanced classes are scheduled throughout the week. The full schedule is available on our mobile app or at the reception desk.",
        metadata={"source": "classes.txt"}
    ),
    Document(
        page_content="Personal trainers at Peak Performance Gym are all certified professionals with minimum 5 years of experience. Each new member receives a complimentary fitness assessment and one free session with a trainer. Our head trainer, Neha Kapoor, specializes in rehabilitation fitness and sports-specific training. Personal training sessions can be booked individually (₹800/session) or in packages of 10 (₹7,000) or 20 (₹13,000).",
        metadata={"source": "trainers.txt"}
    ),
    Document(
        page_content="Peak Performance Gym's facilities include a cardio zone with 30+ machines, strength training area, functional fitness space, dedicated yoga studio, spin class room, swimming pool (25m), sauna and steam rooms, juice bar, and locker rooms with shower facilities. Our equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology.",
        metadata={"source": "facilities.txt"}
    )
]

db = Chroma.from_documents(docs, embedding_function)


class AgentState(TypedDict):
    messages: Annotated[List[str],add_messages]


llm = ChatOpenAI(model='gpt-4o-mini')

def topic_decision(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

def decide(state:AgentState) -> AgentState:
    if True:
        return 'on_topic'
    else:
        return 'off_topic'

def retrieve(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }
 
def generate_answer(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }

def off_topic_reposne(state:AgentState) -> AgentState:
    return {
        "messages":llm.invoke(state['messages'])
    }


graph = StateGraph(AgentState)

graph.set_entry_point('topic_decision')
graph.add_node('topic_decision',topic_decision)
graph.add_node('retrieve',retrieve)
graph.add_node('generate_answer',generate_answer)
graph.add_node('off_topic_response',off_topic_reposne)


graph.add_conditional_edges(
    'topic_decision',
    topic_decision,
    {
        'on_topic':'retrieve',
        'off_topic':'off_topic_response',
    }
)

graph.add_edge('retrieve','generate_answer')
graph.add_edge('generate_answer',END)
graph.add_edge('off_topic_response',END)


app = graph.compile()

app.get_graph().draw_mermaid_png(output_file_path='02_graph.png')