from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Load Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# 1. Connect to your DB
engine  = create_engine(f"mysql+mysqlconnector://sql12811285:x29nsD332f@sql12.freesqldatabase.com:3306/sql12811285")
# Replace with your actual database connection string
db = SQLDatabase(engine )
print("Database connected successfully.")
with engine.connect() as connection:
    result = connection.execute(text("SELECT 1"))
    print(result.fetchall())

# 2. Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"   # or other model names
)


# 3. Build SQL toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.



Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)
agent = create_agent(
    llm,
    tools,
    system_prompt=system_prompt,
)

question = "Can you list down customer details?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values"
):
    step["messages"][-1].pretty_print()

# Clean up
engine.dispose()


# # 2. Initialize Groq LLM
# llm = ChatGroq(
#     groq_api_key=groq_api_key,
#     model_name="mixtral-8x7b"   # or "llama2-70b-4096"
# )

# # 3. Build SQL toolkit
# toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# # 4. Create agent with SQL tools
# agent = create_agent(
#     llm,
#     toolkit.get_tools(),
#     agent="sql_agent",
#     verbose=True
#     )    

# # 5. Run a natural language query
# response = agent.invoke({"input": "Show me the top 5 customers by total purchase amount."})
# print(response)

