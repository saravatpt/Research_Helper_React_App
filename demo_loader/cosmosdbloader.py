from os import environ
from pathlib import Path
from typing import List, Optional, Union
from dotenv import load_dotenv
from pymongo import MongoClient
from jsonloader import JSONLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch, CosmosDBSimilarityType


load_dotenv(override=True)


class CosmosDBLoader():
    def __init__(
    self,
    file_path: Union[str, Path]):
        self.file_path = Path(file_path).resolve()

    def load(self):
        """load embeddings from file_path into cosmosDB vector store"""

        #variable from '.env' file
        AZURE_OPENAI_API_KEY = environ.get("AZURE_OPENAI_API_KEY")
        MONGO_CONNECTION_STRING = environ.get("MONGO_CONNECTION_STRING")

        #hardcoded variables
        DB_NAME = "research"
        COLLECTION_NAME = "resources"
        EMBEDDING_FIELD_NAME = "embedding"
        INDEX_NAME = "vectorSearchIndex"

        client = MongoClient(MONGO_CONNECTION_STRING)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        loader = JSONLoader(self.file_path )

        docs = loader.load()

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 50)
        #docs = text_splitter.split_documents(docs)

        #load documents into Cosmos DB Vector Store
        vector_store = AzureCosmosDBVectorSearch.from_documents(
            docs,
            embedding=AzureOpenAIEmbeddings(
            azure_endpoint="https://myragpocai-resource.openai.azure.com/",
            model="text-embedding-ada-002",
            azure_deployment="text-embedding-ada-002",
            api_key=environ.get("AZURE_OPENAI_API_KEY")
    ),
            collection=collection,
            index_name=INDEX_NAME)        

        if vector_store.index_exists() == False:
            #Create an index for vector search
            num_lists = 1 #for a small demo, you can start with numLists set to 1 to perform a brute-force search across all vectors.
            dimensions = 1536
            similarity_algorithm = CosmosDBSimilarityType.COS

            vector_store.create_index(num_lists, dimensions, similarity_algorithm)


