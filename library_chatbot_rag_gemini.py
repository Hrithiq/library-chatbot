import os
import streamlit as st
import pandas as pd

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_documents():
    df = pd.read_excel("Library_Book_titles_with_dummy_rack_data.xlsx")

    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=(
                    f"Book Title: {row['TITLE']}\n"
                    f"Author: {row['AUTHOR']}\n"
                    f"Aisle: {row['AISLE_NUMBER']}\n"
                    f"Rack: {row['RACK_NUMBER']}\n"
                    f"Section: {row['RACK_SECTION']}\n"
                )
            )
        )
    return docs


# -------------------------------
# VECTOR STORE
# -------------------------------
@st.cache_resource
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(docs, embeddings)


# -------------------------------
# GEMINI RAG CHAIN
# -------------------------------
@st.cache_resource
def build_rag_chain(_vectordb):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key="AIzaSyCjDaHoq-z9wLwJrqCSdRSnQOg4iAJjtH0"
    )

    prompt = ChatPromptTemplate.from_template(
        """
    You are a professional library assistant.

    From the context, extract:
    - Book Title
    - Author
    - Aisle
    - Rack
    - Section

    Respond in this format ONLY:

    Book:
    Author:
    Aisle:
    Rack:
    Section:

    If information is missing, say "Not available".

    Context:
    {context}

    Question:
    {input}
    """
    )


    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = _vectordb.as_retriever(search_kwargs={"k": 4})

    return create_retrieval_chain(retriever, doc_chain)



# -------------------------------
# UI CONFIG
# -------------------------------
st.set_page_config(
    page_title="Library Assistant",
    page_icon="📚",
    layout="centered"
)

st.markdown(
    """
    <style>
    .chat-bubble {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user {
        background-color: #e8f0fe;
    }
    .assistant {
        background-color: #f1f3f4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("📘 Library Assistant")
    st.markdown(
        """
        **AI-powered library search system**

        🔹 Search by book title  
        🔹 Search by author  
        🔹 Instant rack location  

        **Example queries:**
        - Where is *Introduction to Algorithms*?
        - Which rack has books by *Cormen*?
        - Find books written by *George Orwell*

        ---
        **Tech Stack**
        - Gemini 2.5 Flash
        - FAISS Vector Search
        - RAG Architecture
        """
    )

# -------------------------------
# SESSION STATE
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# HEADER
# -------------------------------
st.title("📚 Library Assistant")
st.caption("Ask where a book is located using its title or author")

# -------------------------------
# INPUT
# -------------------------------
query = st.chat_input("Ask a question about a book...")

# -------------------------------
# CHAT LOOP
# -------------------------------
if query:
    st.session_state.chat_history.append(("user", query))

    with st.spinner("Searching library..."):
        docs = load_documents()
        vectordb = build_vectorstore(docs)
        rag_chain = build_rag_chain(vectordb)
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]

    st.session_state.chat_history.append(("assistant", answer))

# -------------------------------
# RENDER CHAT
# -------------------------------
for role, msg in st.session_state.chat_history:
    css_class = "user" if role == "user" else "assistant"
    st.markdown(
        f"<div class='chat-bubble {css_class}'>{msg}</div>",
        unsafe_allow_html=True
    )

