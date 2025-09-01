from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

from typing import TypedDict,List
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from pydantic import BaseModel,Field

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
retriever = db.as_retriever(search_type = 'mmr',search_kwargs={"k":4})

llm = ChatOpenAI(model='gpt-4o-mini')

template = """Answer the question based on the following context and the chathistory. Especially take the latest question into consideration.

Chathistory: {history}
Context: {context}
Question: {quesiton}

"""

prompt = ChatPromptTemplate(template)
  
rag_chain = llm | prompt

class AgentState(TypedDict):
    messages:               List[BaseMessage]
    documents:              List[Document]
    on_topic:               str
    rephrased_question:     str
    proceed_to_generate:    bool
    rephrapse_count:        int
    question:               HumanMessage


class GradeQuestion(BaseModel):
    score: str = Field(description='Qustion is about the specified topics? If Yes -> return "Yes" if Not return "No"')

def question_rewritter(state: AgentState):

    state['document'] =              []
    state['on_topic'] =              ''
    state['rephrased_question'] =    ''
    state['proceed_to_generate'] =   False
    state['rephrapse_count'] =       0

    if 'messages' not in state or state['messages'] is None:
        state['messages'] = []

    if state['question'] not in state['messages']:
        state["messages"].appemd(state['question'])

    if len(state['messages']) > 1:
        conversation     = state['messages'][:-1]
        current_question = state['question'].content

        messages = [
            SystemMessage(
                content = "You are helpful assisatnt that rephrases the user's question to be standalone queation optimized for retrieval"
            )
        ]

        messages.extend(conversation)
        messages.append(
            HumanMessage(content = current_question)
        )

        rephrase_prompt = ChatPromptTemplate.from_messages(messages=messages)
        llm = ChatOpenAI(model='gpt-4o-mini')

        prompt= rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"[question_rewriter] Rephrased qeustion:\n",better_question )
        state['rephrased_question'] = better_question
    else:
        state['rephrased_question'] = state['question'].content
    return state


  