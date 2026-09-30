import streamlit as st

from typing import TypedDict, List

from langchain_core.documents import Document

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_groq import ChatGroq

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from pinecone import Pinecone

from config import (
    GROQ_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    EMBEDDING_MODEL,
    LLM_MODEL
)


# =========================================================
# EMBEDDING MODEL
# =========================================================

embeddings = HuggingFaceEmbeddings(

    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =========================================================
# PINECONE
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(
    PINECONE_INDEX_NAME
)


# =========================================================
# CHATGROQ
# =========================================================

llm = ChatGroq(

    model=LLM_MODEL,

    api_key=GROQ_API_KEY,

    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",

            """
You are a document question-answering assistant.

You MUST follow these rules:

1. Answer ONLY using the provided context.

2. The context comes exclusively from
   the uploaded PDF.

3. Do NOT use your own knowledge.

4. Do NOT make assumptions.

5. Do NOT hallucinate.

6. If the answer is not present in the
   provided context, say:

"I couldn't find the answer to this question
in the provided document."

7. Keep the answer factual and concise.

8. Mention the page number when possible.

CONTEXT:

{context}
"""
        ),

        (
            "human",

            "{question}"
        )
    ]
)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class RAGState(TypedDict):

    question: str

    context: List[Document]

    answer: str

    sources: List[str]


# =========================================================
# RETRIEVE FROM PINECONE
# =========================================================

def retrieve(
    state: RAGState
):

    question = state["question"]


    # -----------------------------------------------------
    # Generate embedding for user question
    # -----------------------------------------------------

    query_vector = embeddings.embed_query(
        question
    )


    # -----------------------------------------------------
    # Search Pinecone
    # -----------------------------------------------------

    results = index.query(

        vector=query_vector,

        top_k=5,

        include_metadata=True,

        namespace=PINECONE_NAMESPACE
    )


    documents = []


    for match in results["matches"]:

        metadata = match.get(
            "metadata",
            {}
        )

        text = metadata.get(
            "text"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )


        if text:

            documents.append(

                Document(

                    page_content=text,

                    metadata={
                        "page": page,

                        "source": metadata.get(
                            "source",
                            "document.pdf"
                        ),

                        "score": match.get(
                            "score",
                            0
                        )
                    }
                )
            )


    return {
        "context": documents
    }


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate(
    state: RAGState
):

    question = state["question"]

    documents = state["context"]


    # -----------------------------------------------------
    # No documents found
    # -----------------------------------------------------

    if not documents:

        return {

            "answer":
                "I couldn't find the answer to this "
                "question in the provided document.",

            "sources": []
        }


    # -----------------------------------------------------
    # Create context
    # -----------------------------------------------------

    context = "\n\n".join(

        [

            f"""
PAGE {doc.metadata.get("page", "Unknown")}

{doc.page_content}
"""

            for doc in documents
        ]
    )


    # -----------------------------------------------------
    # Generate prompt
    # -----------------------------------------------------

    messages = prompt.invoke(

        {
            "context": context,

            "question": question
        }
    )


    # -----------------------------------------------------
    # Call ChatGroq
    # -----------------------------------------------------

    response = llm.invoke(
        messages
    )


    # -----------------------------------------------------
    # Sources
    # -----------------------------------------------------

    sources = []


    for doc in documents:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        source = f"Page {page}"


        if source not in sources:

            sources.append(
                source
            )


    return {

        "answer": response.content,

        "sources": sources
    }


# =========================================================
# BUILD LANGGRAPH
# =========================================================

def create_graph():

    graph = StateGraph(
        RAGState
    )


    graph.add_node(
        "retrieve",
        retrieve
    )


    graph.add_node(
        "generate",
        generate
    )


    graph.add_edge(
        START,
        "retrieve"
    )


    graph.add_edge(
        "retrieve",
        "generate"
    )


    graph.add_edge(
        "generate",
        END
    )


    return graph.compile()


rag_graph = create_graph()


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(

    page_title="PDF RAG Chatbot",

    page_icon="📚",

    layout="wide"
)


st.title(
    "📚 PDF RAG Chatbot"
)


st.caption(
    "Ask questions about the uploaded PDF. "
    "Answers are generated only from the document."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "RAG Pipeline"
    )

    st.write(
        "📄 PDF"
    )

    st.write(
        "✂️ Text Chunking"
    )

    st.write(
        "🧠 HuggingFace Embeddings"
    )

    st.write(
        "🗄️ Pinecone"
    )

    st.write(
        "🔗 LangGraph"
    )

    st.write(
        "🤖 ChatGroq"
    )

    st.divider()


    if st.button(
        "Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if (

            message["role"] == "assistant"

            and message.get("sources")

        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"• {source}"
                    )


# =========================================================
# USER INPUT
# =========================================================

question = st.chat_input(

    "Ask a question about the PDF..."
)


if question:

    # -----------------------------------------------------
    # Display user message
    # -----------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "user",

            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the document..."
        ):

            result = rag_graph.invoke(

                {

                    "question": question,

                    "context": [],

                    "answer": "",

                    "sources": []
                }
            )


        answer = result["answer"]

        sources = result.get(
            "sources",
            []
        )


        st.markdown(
            answer
        )


        # -------------------------------------------------
        # Show sources
        # -------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.write(
                        f"• {source}"
                    )


    # -----------------------------------------------------
    # Save chat
    # -----------------------------------------------------

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer,

            "sources": sources
        }
    )