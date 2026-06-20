# PDF RAG Chatbot 📚🤖

A sophisticated **Retrieval-Augmented Generation (RAG)** system that transforms your PDF documents into an intelligent, conversational knowledge base. Built with FastAPI, LangChain, and a responsive React frontend, completely powered by local LLMs via Ollama[cite: 5, 7, 10].

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🎯 Core Functionality
- **Smart PDF Processing**: Upload multiple PDFs with intelligent text extraction, preprocessing, and chunking[cite: 2, 7, 9].
- **Advanced Document Understanding**: Automatic chapter detection patterns and layout structure analysis[cite: 7].
- **Conversational AI**: Natural language querying of document content with strict point-based professional outputs[cite: 5, 7].
- **Source Attribution**: Every valid response includes document source identifiers and page references[cite: 6, 7].
- **System Quality Assessment**: Built-in confidence scoring thresholds to gauge answer reliability[cite: 3, 7].

### 🎨 Modern UI/UX
- **Glassmorphism Design**: Beautiful, modern interface utilizing custom CSS and Tailwind styling[cite: 10].
- **Interactive Animations**: Smooth transitions and layout animations powered by Framer Motion[cite: 10].
- **Drag & Drop Upload**: Intuitive file upload zone with instant visual feedback states[cite: 10].
- **Full-Screen Layout**: Distraction-free toggle view separating document onboarding and exploration[cite: 10].

---

## 🏗️ Project Architecture

```text
├── .env                  # Local server and RAG settings configurations
├── api.py                # FastAPI endpoint handlers and API logic mapping
├── config.py             # Environment variable parsing into Pydantic Settings
├── exceptions.py         # Custom exception declarations for RAG pipeline steps
├── index.html            # Single-page React app (Frontend)
├── main.py               # Main application entrypoint running Uvicorn
├── models.py             # Pydantic core data schemas for validation
├── rag_system.py         # Core LangChain and FAISS vectorization pipelines
├── run.py                # Preflight check runner script for Ollama dependencies
├── utils.py              # File validations and logging configuration utils
└── requirements.txt      # Python dependencies manifest