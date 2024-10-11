import os
import time
from uuid import uuid4
from .utils import load_pdf, text_split, download_embeddings_model
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_env = os.getenv("PINECONE_API_ENV")

# Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Lazy load embeddings model
embeddings = None

def load_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = download_embeddings_model()
    return embeddings

def connect_vector_db(index_name="medical-assistant-vector-db", dimension=384, add_to_index=False, text_chunks=None):
    '''
    Connect to Pinecone and create an index if necessary.
    Returns: Pinecone vector store
    '''
    try:
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        print(existing_indexes)
        
        if index_name not in existing_indexes:
            print("Creating index")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
        
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=load_embeddings())

        if add_to_index:
            if text_chunks is None:
                raise ValueError("text_chunks must be provided if add_to_index is True")
            add_documents_to_index(vector_store=vector_store, text_chunks=text_chunks)

        return vector_store
    
    except Exception as e:
        raise RuntimeError(f"Error connecting to Pinecone: {str(e)}")

def add_documents_to_index(vector_store, text_chunks, batch_size=100):
    '''
    Add documents to Pinecone in batches.
    '''
    uuids = [str(uuid4()) for _ in range(len(text_chunks))]
    
    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_uuids = uuids[i:i+batch_size]
            vector_store.add_documents(documents=batch, ids=batch_uuids)
    except Exception as e:
        raise RuntimeError(f"Error adding documents to Pinecone: {str(e)}")

def clear_all_records_from_index(vector_store):
    '''
    Clears all records from Pinecone index.
    '''
    try:
        vector_store.delete(delete_all=True)
    except Exception as e:
        raise RuntimeError(f"Error clearing Pinecone index: {str(e)}")


# Connect to Pinecone vector store
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
vector_store = connect_vector_db()
# vector_store = connect_vector_db(add_to_index=True, text_chunks=text_chunks)