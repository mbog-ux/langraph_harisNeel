from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, START, MessagesState
from langgraph.types import Command 
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm              = ChatOpenAI(model='gpt-4o-mini')
tavily_search    = TavilySearch(depth='basic')
python_repl_tool = PythonREPLTool()


class Supervisor(BaseModel):
    next : Literal['enhancer','researcher','coder'] = Field(
        description = "Determines which specialist to activate next in the workflow sequence: "
        "'enchancer' when uset input requires clarification, expansion or refinement, "
        "'researcher' when additional facts, context, or data collection is necessary, "
        "'coder' when implimetation, computation, or technical problem-solving is required"
    )

    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behid selecting the particulr specialist and how this advances the task toward completion"
    )

def supervisor_node(state:MessagesState) -> Command[Literal['enhancer','researcher','coder']]:

    class Supervisor(BaseModel):
        next: Literal["enhancer", "researcher", "coder"] = Field(
            description="Determines which specialist to activate next in the workflow sequence: "
                        "'enhancer' when user input requires clarification, expansion, or refinement, "
                        "'researcher' when additional facts, context, or data collection is necessary, "
                        "'coder' when implementation, computation, or technical problem-solving is required."
        )
        reason: str = Field(
            description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
        )

def supervisor_node(state: MessagesState) -> Command[Literal["enhancer", "researcher", "coder"]]:

    system_prompt = ('''
                 
        You are a workflow supervisor managing a team of three specialized agents: Prompt Enhancer, Researcher, and Coder. Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task. Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.

        **Team Members**:
        1. **Prompt Enhancer**: Always consider this agent first. They clarify ambiguous requests, improve poorly defined queries, and ensure the task is well-structured before deeper processing begins.
        2. **Researcher**: Specializes in information gathering, fact-finding, and collecting relevant data needed to address the user's request.
        3. **Coder**: Focuses on technical implementation, calculations, data analysis, algorithm development, and coding solutions.

        **Your Responsibilities**:
        1. Analyze each user request and agent response for completeness, accuracy, and relevance.
        2. Route the task to the most appropriate agent at each decision point.
        3. Maintain workflow momentum by avoiding redundant agent assignments.
        4. Continue the process until the user's request is fully and satisfactorily resolved.

        Your objective is to create an efficient workflow that leverages each agent's strengths while minimizing unnecessary steps, ultimately delivering complete and accurate solutions to user requests.
                 
    ''')

    messages = [{
        "role":"system",
        "content" : system_prompt
    } + state['messages']]

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto   = response.next
    reason = response.reason

    return Command(
        update ={
            'messages': [HumanMessage(content=reason, name='supervisor')]
        },
        goto = goto
    )

def enhancer_node(state: MessagesState) -> Command[Literal["supervisor"]]:

    """
        Enhancer agent node that improves and clarifies user queries.
        Takes the original user input and transforms it into a more precise,
        actionable request before passing it to the supervisor.
    """
   
    system_prompt = (
        "You are a Query Refinement Specialist with expertise in transforming vague requests into precise instructions. Your responsibilities include:\n\n"
        "1. Analyzing the original query to identify key intent and requirements\n"
        "2. Resolving any ambiguities without requesting additional user input\n"
        "3. Expanding underdeveloped aspects of the query with reasonable assumptions\n"
        "4. Restructuring the query for clarity and actionability\n"
        "5. Ensuring all technical terminology is properly defined in context\n\n"
        "Important: Never ask questions back to the user. Instead, make informed assumptions and create the most comprehensive version of their request possible."
    )

    messages = [
        {"role": "system", "content": system_prompt},  
    ] + state["messages"]  

    enhanced_query = llm.invoke(messages)

    print(f"--- Workflow Transition: Prompt Enhancer → Supervisor ---")

    return Command(
        update={
            "messages": [  
                HumanMessage(
                    content=enhanced_query.content, 
                    name="enhancer"  
                )
            ]
        },
        goto="supervisor", 
    )