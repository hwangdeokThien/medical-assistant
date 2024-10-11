# 🎉 Welcome to **Medical Assistant powered by Llama 2** 👋

**Medical Assistant powered by Llama 2** is an AI-driven application designed to provide users with intelligent medical advice. The assistant processes natural language queries and responds with medical information based on advanced AI models, helping users make informed decisions about their health.
<img width="1440" alt="med_ast_demo" src="https://github.com/user-attachments/assets/db266343-6ae5-448f-b6c6-55fc870bb966">

## 🌟 Key Features:
- AI-powered medical advice and responses.
- Efficient data retrieval using embedding-based search.
- User-friendly interface and intuitive conversation flow.

## 💻 Tech Stack:
- **LangChain** - For building the conversational AI pipeline.
- **Pinecone** - Manages and searches embeddings for fast, accurate results.
- **Llama 2** - Powers the natural language understanding and response generation.
- **Flask** - A lightweight framework to manage the backend and API.

---
## 🚀 Getting Started -- Running the Application Locally

### Installing LLMs model
1. Install the `llama-2-7b-chat.Q4_0.gguf` model from [this link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/tree/main).

2. Move the downloaded file into `./model/` directory.

### Running the Flask app

1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the app**:
    ```bash
    flask run
    ```
---

### Additional Configuration
Make sure to configure your environment variables as outlined in the `env_template` file before running the project.

## 🛠️ Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have ideas or improvements.

## 📧 Contact

For any questions or support, please reach out to [huynhducthien41906@gmail.com](mailto:huynhducthien41906@gmail.com).

---

Enjoy using **Medical Assistant powered by Llama 2**! ✨
