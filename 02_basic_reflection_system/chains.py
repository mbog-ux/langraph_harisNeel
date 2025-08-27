from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " You are a twitter technie imfluencer assistance tasked with writting excelent twitter posts."
            " Generete the best twitter post possible for the users's request."
            " if the user provides critiques, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are viral twiter influencer granding a atweet. Generate ctritique and recomendations for the user's twee."
            "Alway provide detailed recommendation, including requests for length, virality, style, etv."
        ),
        MessagesPlaceholder(variable_name='messages')

    ]
)

llm = ChatOpenAI(model='gpt-4o-mini')

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# from langchain.chains import LLMChain
# chain = LLMChain(prompt=generation_prompt, llm=llm)


