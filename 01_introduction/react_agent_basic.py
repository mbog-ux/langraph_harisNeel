from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain.agents import initialize_agent,tool
from langchain_community.tools import TavilySearchResults
import datetime

load_dotenv()

llm = ChatOpenAI(model = 'gpt-4o-mini',temperature=0)

search_tool = TavilySearchResults(search_depth = 'basic')

@tool
def get_system_time(format: str = '%Y-%m-%d %H:%M:%S'):
    "Returns the current date and time in the specified format"
    curent_time= datetime.datetime.now()
    formated_time = curent_time.strftime(format)
    return formated_time

tools = [search_tool,get_system_time]

agent = initialize_agent(tools =tools,llm = llm, agent="zero-shot-react-description",verbose=True)
agent.invoke('When was SpaceX the latest launch and how many days ago was it from today?')