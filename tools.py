from langchain.tools import BaseTool
from db import VectorDB
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import PromptTemplate

import requests

class SymbolFetchTool(BaseTool):
    name: str = "company_symbol_fetch"
    description: str = "Fetches symbol for specific company or corporation name."
    api_key: str = ''

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _run(self, company_name: str) -> str:
        # API endpoint for fetching weather
        url = f"https://financialmodelingprep.com/api/v3/search?query={company_name}&apikey={self.api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return f"{data}"
        else:
            return "Could not find company based on name."

     
class FMPAPITool(BaseTool):
    name: str = "financial_api"
    description: str = """Fetches information based on retrieved urls with params set based on original question.
    Takes the formatted result of the url along with the company name and the query parameters inferred from the original user input,
    following the parameters format of the url. """
    api_key: str = ''

    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def _run(self, url: str) -> str:
        # API endpoint for fetching weather
        url = f"{url}&apikey={self.api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return f"{data}"
        else:
            return """Could not find additional information relevant to company based on api calls.
            This could be due to """
            
            
class DBTool():
    
    def get_tool():
        #tool only retrieves one url from DB based on similarity, once
        # DB is filled, this value should be increased.
        retriever = VectorDB.db.as_retriever(search_kwargs={"k": 1}, search_type="similarity")  
        description = """Use to look up which urls to use for retrieval of financial related information, based on original question"""
        return create_retriever_tool(
            retriever,
            name="search_relevant_urls",
            description=description,
            document_prompt=PromptTemplate.from_template(" URL: {page_content}\n Examples of url use with params: {examples}")
        ) 