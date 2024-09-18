from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()
chat = ChatOpenAI()
table_names = list_tables()
# print(tables)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = (
            "You are an AI that has access to SQLimte database.\n"
            f"The database has following tables: {table_names}\n"
            " Do not make any assumptions about what tables exists or what columns exists "
            "Instead use the 'describe_tables' function" 
        )),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

tools = [run_query_tool, describe_tables_tool]

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