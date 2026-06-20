from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging

from config import settings
from models import ChatMessage, ChatResponse, UploadResponse, StatusResponse, ResetResponse
from api import PDFChatbotAPI
from utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PDF RAG Chatbot API",
    description="A chatbot that answers questions based on uploaded PDF documents using RAG",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API handler
api_handler = PDFChatbotAPI()

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await api_handler.health_check()

@app.post("/upload-pdfs", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    return await api_handler.upload_pdfs(files)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat with the PDF-powered chatbot"""
    return await api_handler.chat(message)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    return await api_handler.get_status()

@app.delete("/reset", response_model=ResetResponse)
async def reset_system():
    """Reset the RAG system (clear all documents)"""
    return await api_handler.reset_system()

@app.get("/info")
async def get_system_info():
    """Get detailed system information"""
    return api_handler.rag_system.get_system_info()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("PDF RAG Chatbot API starting up...")
    logger.info(f"Ollama URL: {settings.ollama_base_url}")
    logger.info(f"Model: {settings.ollama_model}")
    logger.info(f"Embedding model: {settings.embedding_model}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("PDF RAG Chatbot API shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.api_host, 
        port=settings.api_port,
        log_level="info"
    )
