import os

from dotenv import load_dotenv


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "rag-chatbot"
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "pdf-documents"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-en-v1.5"
)

PDF_PATH = "data/document.pdf"


if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing in .env"
    )


if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing in .env"
    )