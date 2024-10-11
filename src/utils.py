from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data):
   '''
   Function for loading PDF files
   data: path to directory containing PDF files
   returns: list of documents
   '''
   loader =  DirectoryLoader(path=data, 
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
   documents = loader.load()

   return documents

def text_split(extracted_data):
    '''
    Function for splitting text to chunks
    extracted_data: list of documents
    returns: list of text chunks
    '''
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents=extracted_data)
    
    return text_chunks

def download_embeddings_model():
    '''
    Function for downloading embeddings model from HuggingFace
    returns: embeddings model
    '''
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings