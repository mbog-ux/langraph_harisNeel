from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
from langchain_openai import ChatOpenAI
from schema import AnswerQuestion,ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
pyndatic_parser = PydanticToolsParser(tools = [AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are exeprt AI researcher.
            Current time: {time}

            1. {first_instruction}
            2. Reflect and critiques your answer. Be severe to maximize improvment.
            3. After the reflection, **list 1-3 search queries separetly** for researing improvements. Do not include them inside the reflection.
            """
        ),
        MessagesPlaceholder(variable_name='messages'),
        (
            "system","Answer the user's question about using the required format."
        )
    ]
).partial(time = datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction = "Provide a detailed ~250 words answer"
    )

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools = [AnswerQuestion], tool_choice='AnswerQuestion') 
validator = PydanticToolsParser(tools = [AnswerQuestion])


# Revisor section

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

response = first_responder_chain.invoke({
    "messages":[HumanMessage(content = 'Write a blog post on how small buisness can leverage AI to grow')]
})