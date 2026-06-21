# PDF RAG Chatbot 📚🤖

A sophisticated **Retrieval-Augmented Generation (RAG)** system that transforms your PDF documents into an intelligent, conversational knowledge base. Built with modern web technologies and powered by local LLMs via Ollama.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🎯 Core Functionality
- **Smart PDF Processing**: Upload multiple PDFs with intelligent text extraction and chunking
- **Advanced Document Understanding**: Automatic chapter detection and structure analysis
- **Conversational AI**: Natural language querying of document content
- **Structured Responses**: Professional, bullet-pointed answers with source citations
- **Real-time Chat**: Smooth, responsive chat interface with typing indicators

### 🧠 AI & RAG Features
- **Local LLM Integration**: Powered by Ollama (Mistral 7B by default)
- **Semantic Search**: Advanced embeddings with FAISS vector store
- **Context-Aware Responses**: Retrieves relevant chunks with confidence scoring
- **Source Attribution**: Every response includes document sources and page references
- **Quality Assessment**: Built-in confidence scoring for answer reliability

### 🎨 Modern UI/UX
- **Glassmorphism Design**: Beautiful, modern interface with blur effects
- **Responsive Layout**: Works seamlessly on desktop and mobile
- **Interactive Animations**: Smooth transitions and hover effects powered by Framer Motion
- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Full-Screen Chat**: Distraction-free chat experience

## 🏗️ Architecture

```text
├── .env                  # Local server and RAG settings configurations
├── api.py                # FastAPI endpoint handlers and API logic mapping
├── config.py             # Environment variable parsing into Pydantic Settings
├── exceptions.py         # Custom exception classes
├── index.html            # Single-page React app (Frontend)
├── main.py               # Main application entrypoint running Uvicorn
├── models.py             # Pydantic core data schemas for validation
├── rag_system.py         # Core LangChain and FAISS vectorization pipelines
├── run.py                # Preflight check runner script for Ollama dependencies
├── utils.py              # File validations and logging configuration utils
└── requirements.txt      # Python dependencies manifest