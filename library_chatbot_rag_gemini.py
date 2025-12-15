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
def load_documents_from_df(df):
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
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(
        """
    You are a professional library assistant.
    
    From the context, identify ALL relevant books.
    Return ONE entry per book.

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
    (each in a new line)

    If multiple books match, list multiple blocks.
    If no book matches, say: "No matching books found."

    Context:
    {context}

    Question:
    {input}
    """
    )


    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = _vectordb.as_retriever(search_kwargs={"k": 8})

    return create_retrieval_chain(retriever, doc_chain)

# -------------------------------
# RENDER MULTI-BOOK RESULTS
# -------------------------------
def render_books(answer_text):
    books = answer_text.split("---")
    for book in books:
        if book.strip():
            formatted_book = book.replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="border:1px solid #ddd;
                            padding:12px;
                            border-radius:8px;
                            margin-bottom:10px;">
                    {formatted_book}
                </div>
                """,
                unsafe_allow_html=True
            )


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
        - Where is *Essentials of Physical Chemistry*?
        - Which rack has books by *Jones Joy*?
        - Find books written by *Hubbard Ron L*

        ---
        **Tech Stack**
        - FAISS Vector Search
        - RAG Architecture
        """
    )

    st.markdown("---")
    st.subheader("🔐 Admin Panel")

    uploaded_file = st.file_uploader(
        "Upload Library Excel",
        type=["xlsx"]
    )


# -------------------------------
# SESSION STATE
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "library_df" not in st.session_state:
    st.session_state.library_df = None

if uploaded_file:
    st.session_state.library_df = pd.read_excel(uploaded_file)
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Library data uploaded and indexed")


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
        if st.session_state.library_df is not None:
            docs = load_documents_from_df(st.session_state.library_df)
            st.cache_resource.clear()
            st.cache_data.clear()

        else:
            df = pd.read_excel("Library_Book_titles_with_dummy_rack_data.xlsx")
            docs = load_documents_from_df(df)

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
    if role == "assistant":
        render_books(msg)
    else:
        st.markdown(
            f"<div class='chat-bubble {css_class}'>{msg}</div>",
            unsafe_allow_html=True
        )

