from .utils import load_pdf, download_embeddings_model, text_split
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from uuid import uuid4
import time
import os

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_env = os.getenv("PINECONE_API_ENV")

pc = Pinecone(api_key=pinecone_api_key)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_embeddings_model()


# connect to index
index_name = 'medical-assistant-vector-db'
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# clear all the current docs, restore the space
# index.delete(delete_all=True) # we can set the specific name space as well

# add documents to vector db
uuids = [str(uuid4()) for _ in range(len(text_chunks))]
vector_store.add_documents(documents=text_chunks, ids=uuids)