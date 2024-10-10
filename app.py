from flask import Flask, render_template, jsonify, request
from langchain.llms import CTransformers
from src.vector_store import vector_store
from langchain.chains import RetrievalQA
from src.prompt import prompt_template
from langchain.prompts import PromptTemplate

app = Flask(__name__)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.Q4_0.gguf",
    model_type="llama",
    config={'max_new_tokens':512, 'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


### flask app
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)