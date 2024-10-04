from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

class VectorDB():

    db = Chroma(embedding_function=OpenAIEmbeddings(api_key=api_key), collection_name="Finchat")


    def insert_into_db(key, metadata={}):
        new_doc = Document(
            page_content=key,
            metadata=metadata
        )
        VectorDB.db.add_documents([new_doc])


    def search_db(key, *args):
        embedding_vector = OpenAIEmbeddings(api_key=api_key).embed_query(key)
        docs = VectorDB.db.similarity_search_by_vector(embedding_vector, *args)
        return docs
    


#add all url endpoints of interest in DB
#for the sake of this exercise added two only    

vector_db = VectorDB()
VectorDB.insert_into_db(
    'https://financialmodelingprep.com/api/v3/income-statement/',
    metadata = {'examples': str([
        'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=annual',
        'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter'    
    ])}
)
VectorDB.insert_into_db(
    'https://financialmodelingprep.com/api/v3/earning_call_transcript/',
    metadata = {'examples': str([
        'https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?year=2020&quarter=2',
    ])}
)