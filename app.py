from flask import Flask, render_template, jsonify, request
from langchain.llms import CTransformers
from src.vector_store import vector_store
from src.prompt import prompt_template
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


app = Flask(__name__)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.Q4_0.gguf",
    model_type="llama",
    config={'max_new_tokens':512, 'temperature':0.8})

retriever = vector_store.as_retriever(search_kwargs={'k': 2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

### flask app
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(input)
    print("Response: ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)