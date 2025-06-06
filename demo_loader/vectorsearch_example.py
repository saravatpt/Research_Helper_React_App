from os import environ
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

load_dotenv(override=True)

#variable from '.env' file
MONGO_CONNECTION_STRING = environ.get("MONGO_CONNECTION_STRING")

#hardcoded variables
DB_NAME = "research"
COLLECTION_NAME = "resources"
INDEX_NAME = "vectorSearchIndex"

#connect to Azure Cosmos DB for vector search
vector_store = AzureCosmosDBVectorSearch.from_connection_string(
    connection_string=MONGO_CONNECTION_STRING,
    namespace=DB_NAME + "." + COLLECTION_NAME,
    embedding=AzureOpenAIEmbeddings(
        azure_endpoint="https://myragpocai-resource.openai.azure.com/",
        model="text-embedding-ada-002",
        azure_deployment="text-embedding-ada-002",
        api_key=environ.get("AZURE_OPENAI_API_KEY")
    ),
    index_name=INDEX_NAME
)

#query to use in similarity_search
query = "supersonic combustion"

docs = vector_store.similarity_search(query,k=10)

#format results from search
for doc in docs:
    print({'id' :doc.metadata["page_id"],
           'title':doc.metadata["title"],           
            'source':f"{doc.metadata['chapter']}  (page-{doc.metadata['pagenumber']})",                        
            'content':doc.page_content})
