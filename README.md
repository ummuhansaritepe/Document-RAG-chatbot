# Document RAG Chatbot

A Retrieval-Augmented Generation (RAG) chat application built with Streamlit that lets you upload PDF, TXT, CSV, and Excel files and ask questions about their content.

## Features

- Upload and process multiple files at once (PDF, TXT, CSV, XLSX)
- Splits documents into chunks and stores them as embeddings in a vector store
- Retrieves the most relevant chunks for a question and generates an answer based on them
- Keeps chat history during the session

## Installation

1. Clone the repository and move into the project folder:

   ```bash
   git clone <repo-url>
   cd document-rag-chatbot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file and add the required API keys (example):

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the App

```bash
streamlit run app.py
```

By default, the app runs at `http://localhost:8501`.

## Usage

1. Upload one or more files (PDF, TXT, CSV, XLSX) using the file uploader.
2. Click **Process documents** to read the files and build embeddings.
3. Type your question about the documents into the chat input at the bottom.
4. The app retrieves the most relevant sections and generates an answer.

## Project Structure

```
.
├── app.py            # Streamlit UI
├── rag.py            # Chunking, vector store, retrieval, and answer generation logic
├── requirements.txt  # Required dependencies
├── .env              # API keys (not shared)
└── README.md
```

## License

This project is intended for a company.  
