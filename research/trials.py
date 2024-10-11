from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import os
import time
from dotenv import load_dotenv

load_dotenv()

def download_embeddings_model():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embeddings = download_embeddings_model()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_api_env = os.getenv("PINECONE_API_ENV")

pc = Pinecone(api_key=pinecone_api_key)

index_name = 'medical-assistant-vector-db'

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
# uuids = [str(uuid4()) for _ in range(len(text_chunks))]
# vector_store.add_documents(documents=text_chunks, ids=uuids)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vector_store.as_retriever(search_kwargs={'k': 2})

model_path = os.path.abspath("/Users/hwangdeokthien/Code/project/medical-assistant/model/llama-2-7b-chat.Q4_0.gguf")
llm = CTransformers(model=model_path,
                    model_type="llama",
                    config={'max_new_tokens':512, 'temperature':0.8})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

print('oke')
response = rag_chain.invoke('What are Allergies?')
print("Response: ", response)