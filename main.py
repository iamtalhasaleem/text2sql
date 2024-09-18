from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables

load_dotenv()
chat = ChatOpenAI()
tables = list_tables()
# print(tables)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = 'You are an AI that has access to SQLimte database.\n{tables}'),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_query_tool]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(
    agent= agent,
    verbose= True,
    tools= tools
)

agent_executor("How many users have provided the shipping address ?")