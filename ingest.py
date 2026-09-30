
from pathlib import Path
import uuid

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from pinecone import (
    Pinecone,
    ServerlessSpec
)

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    PDF_PATH
)

# CONFIGURATION

EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 800

CHUNK_OVERLAP = 150


# EMBEDDING MODEL

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# LOAD PDF

def load_pdf():

    if not Path(PDF_PATH).exists():

        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    reader = PdfReader(
        PDF_PATH
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": PDF_PATH,
                        "page": page_number + 1
                    }
                )
            )

    return documents


# CHUNK DOCUMENT

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# CONNECT TO PINECONE

def get_pinecone_index():

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    existing_indexes = [
        index["name"]
        for index in pc.list_indexes()
    ]

    if PINECONE_INDEX_NAME not in existing_indexes:

        print(
            "Creating Pinecone index..."
        )

        pc.create_index(

            name=PINECONE_INDEX_NAME,

            dimension=EMBEDDING_DIMENSION,

            metric="cosine",

            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(
        PINECONE_INDEX_NAME
    )

# CREATE EMBEDDINGS + STORE IN PINECONE

def store_documents(chunks):

    index = get_pinecone_index()

    print(
        "Generating embeddings..."
    )

    vectors = []

    for chunk in chunks:

        # Generate embedding for the chunk
        vector = embeddings.embed_query(
            chunk.page_content
        )

        vector_id = str(
            uuid.uuid4()
        )

        vectors.append(
            {
                "id": vector_id,

                "values": vector,

                "metadata": {
                    "text": chunk.page_content,

                    "source": chunk.metadata.get(
                        "source",
                        PDF_PATH
                    ),

                    "page": chunk.metadata.get(
                        "page",
                        0
                    )
                }
            }
        )

    print(
        f"Generated {len(vectors)} embeddings."
    )

    # Upload to Pinecone
    index.upsert(
        vectors=vectors,
        namespace=PINECONE_NAMESPACE
    )

    print(
        f"Stored {len(vectors)} vectors in Pinecone."
    )


# MAIN INGESTION PIPELINE

def ingest():

    print(
        "\n========== PDF INGESTION ==========\n"
    )

    # 1. Load PDF
    print("1. Loading PDF...")

    documents = load_pdf()

    print(
        f"   Loaded {len(documents)} pages."
    )


    # 2. Chunk PDF
    print("\n2. Splitting document...")

    chunks = split_documents(
        documents
    )

    print(
        f"   Created {len(chunks)} chunks."
    )


    # 3. Generate embeddings
    print(
        "\n3. Generating embeddings..."
    )


    # 4. Store in Pinecone
    print(
        "\n4. Storing vectors in Pinecone..."
    )

    store_documents(
        chunks
    )


    print(
        "\n=== INGESTION COMPLETE ===\n"
    )


if __name__ == "__main__":

    ingest()