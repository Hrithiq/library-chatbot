# 📚 Library Assistant Chatbot (Gemini + RAG)

An AI-powered **Library Search Assistant** that allows users to locate books in a physical library using **text or voice queries**.

Built using **Retrieval-Augmented Generation (RAG)** with **Google Gemini**, **FAISS vector search**, and a **Streamlit-based chat UI**.

---

## 🚀 Features

### 🔍 Intelligent Book Search
- Search by **Book Title**
- Search by **Author Name**
- Supports **multiple book matches**

### 🎙️ Voice + Text Input
- Type queries or speak naturally
- Automatic speech-to-text conversion
- Noise-tolerant preprocessing

### 🧠 RAG Architecture
- Excel dataset → embeddings → FAISS vector DB
- Gemini LLM answers strictly from retrieved context
- Prevents hallucination

### 🗂️ Admin Panel
- Upload new Excel library datasets at runtime
- Automatic re-indexing
- No restart required

### 💬 Chat-Based Interface
- Persistent conversation history
- Clean, library-grade UI
- Multi-book results rendered as structured cards

---

## 🏗️ System Architecture

