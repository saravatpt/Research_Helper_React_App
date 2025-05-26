from os import environ
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch


load_dotenv(override=True)


collection: Collection | None = None
vector_store: AzureCosmosDBVectorSearch | None=None

def mongodb_init():
    MONGO_CONNECTION_STRING = environ.get("MONGO_CONNECTION_STRING")
    DB_NAME = "research"
    COLLECTION_NAME = "resources"
    INDEX_NAME = "vectorSearchIndex"

    global collection, vector_store
    client = MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    vector_store = AzureCosmosDBVectorSearch.from_connection_string(MONGO_CONNECTION_STRING,
                                                                    DB_NAME + "." + COLLECTION_NAME,
                                                                        embedding=AzureOpenAIEmbeddings(
        azure_endpoint="https://myragpocai-resource.openai.azure.com/",
        model="text-embedding-ada-002",
        azure_deployment="text-embedding-ada-002",
        api_key=environ.get("AZURE_OPENAI_API_KEY")
    ),
                                                                    index_name=INDEX_NAME                                                                    
)
   

mongodb_init()

