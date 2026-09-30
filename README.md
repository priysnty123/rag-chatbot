# PDF RAG Chatbot

A Streamlit-based retrieval-augmented generation (RAG) chatbot that answers questions using the contents of a local PDF. The application retrieves relevant PDF chunks from Pinecone and uses Groq to generate concise answers with page references.

## Features

- PDF ingestion from `data/document.pdf`
- Recursive text chunking with overlap
- Local Hugging Face embeddings
- Pinecone vector search
- Groq-powered answer generation
- LangGraph workflow for retrieval and generation
- Streamlit chat interface with source pages

## Prerequisites

- Python 3.10 or newer
- A [Pinecone](https://www.pinecone.io/) account and API key
- A [Groq](https://console.groq.com/) API key
- Internet access for model downloads and hosted API calls

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root. Do not commit this file because it contains credentials.

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Optional: these values have defaults in config.py
PINECONE_INDEX_NAME=rag-chatbot
PINECONE_NAMESPACE=pdf-documents
LLM_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

The Pinecone index is created automatically by `ingest.py` if it does not already exist. It uses cosine similarity, dimension `768`, and the AWS `us-east-1` serverless region.

### 4. Add the source PDF

Place the document you want to query at:

```text
data/document.pdf
```

The current implementation expects this exact path and filename.

### 5. Ingest the PDF

Run ingestion once before starting the chatbot, or rerun it whenever the source PDF changes:

```bash
python ingest.py
```

This loads the PDF, splits it into chunks, generates normalized embeddings, and stores the vectors and page metadata in Pinecone.

### 6. Start the chatbot

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`, and ask questions about the indexed PDF.

## Usage

1. Ensure the PDF has been ingested into the configured Pinecone index and namespace.
2. Open the Streamlit application.
3. Enter a question in the chat box.
4. Review the answer and expand **Sources** to see the pages used for retrieval.
5. Use **Clear Chat** in the sidebar to reset the conversation display.

The prompt instructs the model to answer only from retrieved document context. When the retrieved context does not contain an answer, the chatbot returns:

> I couldn't find the answer to this question in the provided document.

## Architecture

```text
                    Ingestion (offline)

 data/document.pdf
        |
        v
  pypdf page extraction
        |
        v
 RecursiveCharacterTextSplitter
        |
        v
 Hugging Face embeddings
        |
        v
 Pinecone index + page metadata

                    Question answering (Streamlit)

 User question
        |
        v
 Hugging Face query embedding
        |
        v
 Pinecone similarity search (top 5 chunks)
        |
        v
 LangGraph: retrieve -> generate
        |
        v
 Groq LLM with retrieved context
        |
        v
 Answer + source page numbers
```

### Main components

- `ingest.py`: extracts PDF text, creates overlapping chunks, embeds them, and writes vectors to Pinecone.
- `app.py`: provides the Streamlit chat UI and runs the LangGraph retrieval/generation workflow.
- `config.py`: loads credentials and model/index settings from `.env`.
- `data/document.pdf`: source document used by the ingestion pipeline.

## Configuration notes

- The embedding model produces 768-dimensional vectors; the Pinecone index dimension must remain `768` unless the ingestion code is updated.
- The app and ingestion script must use the same `PINECONE_INDEX_NAME` and `PINECONE_NAMESPACE`.
- Re-ingesting the same PDF creates new UUID-based vectors. Use a fresh namespace or clear old vectors if you need to avoid duplicate records.
- API keys are validated when `config.py` is imported.

## Project structure

```text
.
├── app.py
├── config.py
├── ingest.py
├── requirements.txt
├── data/
│   └── document.pdf
└── .env
```
