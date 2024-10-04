from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
import os, datetime

from tools import SymbolFetchTool, FMPAPITool, DBTool

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


class Agent():
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an assistant whose  objective is to detect
                and extract company or corporate entities' names from the user input,
                retrieve base urls relevant to user question, determine time ranges or variables based on user question, build final urls based on human question, url examples and time ranges determined,
                and then summarize information provided by api calls to answer question.
                If no precise date is specified in user question, consider the date as the current one: {datetime.datetime.now()}
                """,
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    def __init__(self):
        #create tool calling agent, and pass to it 
        # SymbolFetchTool to convert company names to symbols
        # DBtool to get relevant urls for retrieval augmentation
        # FMPAPITool to access FMP api with reformatted urls
        agent = create_tool_calling_agent(
            tools= [
                SymbolFetchTool(api_key=os.getenv("FMP_KEY")),
                DBTool.get_tool(),
                FMPAPITool(api_key=os.getenv("FMP_KEY"))
            ],
            llm=llm,
            prompt=Agent.prompt,
        )
        self.executor = AgentExecutor(
            agent=agent,
            tools=[
                SymbolFetchTool(api_key=os.getenv("FMP_KEY")),
                DBTool.get_tool(),
                FMPAPITool(api_key=os.getenv("FMP_KEY"))
            ],
            verbose=True
        )


