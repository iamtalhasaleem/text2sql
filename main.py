from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler
load_dotenv()


handler = ChatModelStartHandler()
chat = ChatOpenAI(
        callbacks = [handler]
    )

table_names = list_tables()
# print(tables)

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content = (
            "You are an AI that has access to SQLite database.\n"
            f"The database has following tables: {table_names}\n"
            " Do not make any assumptions about what tables exists or what columns exists "
            "Instead use the 'describe_tables' function" 
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
        
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages= True)

tools = [
    run_query_tool, 
    describe_tables_tool,
    write_report_tool
    ]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(
    agent = agent,
    # verbose = True,
    tools = tools,
    memory = memory
)

# agent_executor("How many users have provided the shipping address ?")
# agent_executor("Summerize the top 5 most popular products. Write the results to a report file")

# agent_executor("How many orders are there. Write the results to a report")
# agent_executor("Repeat exact process for users")

agent_executor("How many users are there. Write the results to a report")