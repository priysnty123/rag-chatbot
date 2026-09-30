# PDF RAG Chatbot     Link = https://rag-assignment-9129.streamlit.app/

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

# Optional: these values which I have set in config.py
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

## Sample Question

## Question 1 : what does pdf is talking about ?

Answer : The PDF discusses Agentic AI and multi‑agent systems, covering their core concepts, architecture, and practical applications:

Definition and scope of Agentic AI – explains what Agentic AI is, how it differs from other AI, its capabilities, value, and real‑world business use cases (Page 7.0).
Key components of autonomous agents – autonomy, goal‑driven behavior, long‑term memory, and the Beliefs‑Desires‑Intentions (BDI) model that characterizes agents’ internal state (Pages 21.0 & 22.0).
Social ability & communication – agents communicate via well‑defined protocols (e.g., Model Context Protocol) to exchange requests, assertions, and queries, enabling negotiation, coordination, and cooperation; natural‑language communication is highlighted with LLMs (Pages 22.0 & 40.0).
Orchestrating multi‑agent systems – outlines the importance of efficient communication mechanisms and protocols for successful collaboration among agents (Page 40.0).
Authorship and contributors – lists the senior author, co‑authoring team, designers, and marketing lead involved in creating the book (Page 59.0).
Overall, the document provides a comprehensive overview of how autonomous, communicative agents are designed, how they interact, and how they are applied in emerging AI solutions.

Confidence score: 0.4832


## Question 2: What are Categories and Types of Agentic Systems ?

Answer: Categories of Agentic Systems

Based on complexity and application – Agentic AI can be grouped by how sophisticated they are and where they are applied. Examples given include Simple Reflex Agents (which react instantly to a trigger without any internal model) and Model‑Based Agents (which maintain an internal representation of the world to guide their actions)【PAGE 23.0】.

Based on functionality and behavior – The same two categories (Simple Reflex and Model‑Based) are also described from the viewpoint of what the agents can do and how they behave【PAGE 23.0】.

Single‑agent vs. multi‑agent systems – A single‑agent architecture handles well‑defined, systematic tasks, while Multi‑Agent Systems (MAS) involve several agents that can give each other feedback and execute tasks in parallel【PAGE 30.0】.

Types of Agents (functional versatility)

Reactive Agents – Respond quickly to immediate stimuli without long‑term planning (e.g., a real‑time data‑anomaly detector)【PAGE 24.0】.
Deliberative Agents – Perform planning by considering multiple variables and possible outcomes (e.g., predictive‑maintenance AI)【PAGE 24.0】.
Hybrid Agents – Combine reactive and deliberative capabilities for dynamic, complex scenarios【PAGE 24.0】.
Types of Atomic Agents (a finer‑grained classification)

Foundational Agents – Provide core capabilities such as planning or verification that support other agents (e.g., a planner that optimizes production schedules)【PAGE 25.0】.
Workflow Agents – Execute specific high‑quality tasks in a vertical domain (e.g., a coding or web‑navigation agent)【PAGE 25.0】.
Utility Agents – Offer specialized utilities that can be combined with other agents to extend functionality【PAGE 25.0】.
Together, these categories and types describe the spectrum of agentic systems—from simple, reflexive bots to complex, collaborative multi‑agent architectures with specialized functional roles.

Confidence score: 0.8307


## Question 3: What is Industry-Wise Readiness ?

Answer: Industry‑Wise Readiness is the process of evaluating an organization’s preparedness for Agentic AI across a set of key parameters—such as data infrastructure, primary AI use cases, data quality, technological maturity, skill readiness, regulatory constraints, ethical considerations, and IoT sensors—tailored to each specific industry. This checklist helps firms identify strengths, uncover gaps, and determine what foundational elements need to be addressed before moving forward with Agentic AI adoption. (Page 50)

Confidence score: 0.7630

## Question 4: Who is prime minister of India ?

Answer :I couldn't find the answer to this question in the provided document.




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
