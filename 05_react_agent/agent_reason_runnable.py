from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain_community.tools import TavilySearchResults
from langchain import hub

llm = ChatOpenAI(model = 'gpt-4o-mini')

search_tool = TavilySearchResults(search_depth = 'basic')

@tool
def get_system_time(format: str = '%Y-%m-%d %H:%M:%S'):
    "Returns the current date and time in the specified format"
    curent_time= datetime.datetime.now()
    formated_time = curent_time.strftime(format)
    return formated_time

tools = [search_tool,get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools = tools,llm=llm, prompt=react_prompt)