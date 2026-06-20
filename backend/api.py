"""Enhanced API with improved error handling and additional features"""
from fastapi import HTTPException, UploadFile, File, Query
from typing import List, Optional
import os
import tempfile
import logging
from pathlib import Path

from models import ChatMessage, ChatResponse, UploadResponse, StatusResponse, ResetResponse
from rag_system import EnhancedPDFRAGSystem  # Fixed import name
from utils import validate_pdf_files, cleanup_temp_files
from exceptions import PDFProcessingError, RAGSystemNotReadyError

logger = logging.getLogger(__name__)

class PDFChatbotAPI:
    """Enhanced API handler class with better context handling"""
    
    def __init__(self):
        self.rag_system = EnhancedPDFRAGSystem()  # Fixed class name
    
    async def health_check(self):
        """Enhanced health check with system info"""
        ollama_status = await self.rag_system.check_ollama_status()
        system_info = self.rag_system.get_system_info()
        
        return {
            "status": "healthy",
            "message": "Enhanced PDF RAG Chatbot API is running",
            "ollama_available": ollama_status,
            "system_ready": system_info["ready"],
            "documents_loaded": system_info["documents_loaded"]
        }
    
    async def upload_pdfs(self, files: List[UploadFile]) -> UploadResponse:
        """Enhanced PDF upload with better error handling and feedback"""
        try:
            if not files:
                raise HTTPException(status_code=400, detail="No files provided")
            
            # Validate file types and sizes
            validate_pdf_files(files)
            
            # Check for reasonable file sizes
            total_size = 0
            for file in files:
                file_size = 0
                content = await file.read()
                file_size = len(content)
                total_size += file_size
                await file.seek(0)  # Reset file pointer
                
                # Check individual file size (50MB limit)
                if file_size > 50 * 1024 * 1024:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"File {file.filename} is too large. Maximum size is 50MB."
                    )
            
            # Check total size (200MB limit)
            if total_size > 200 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="Total upload size too large. Maximum is 200MB."
                )
            
            # Save uploaded files temporarily and process them
            temp_files = []
            try:
                for file in files:
                    # Create temporary file with proper naming
                    file_suffix = Path(file.filename).suffix if file.filename else '.pdf'
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix)
                    temp_files.append(temp_file.name)
                    
                    # Write uploaded content to temp file
                    await file.seek(0)  # Ensure we're at the start
                    content = await file.read()
                    temp_file.write(content)
                    temp_file.close()
                    
                    logger.info(f"Saved {file.filename} to temporary file {temp_file.name}")
                
                # Process PDFs with enhanced RAG system
                result = await self.rag_system.process_pdfs(temp_files)
                
                logger.info(f"Successfully processed {len(files)} PDF files")
                
                return UploadResponse(
                    message=f"Successfully processed {len(files)} PDF files with {result.get('chunks_created', 0)} chunks",
                    files_processed=[file.filename for file in files],
                    chunks_created=result.get("chunks_created", 0)
                )
                
            finally:
                # Clean up temporary files
                cleanup_temp_files(temp_files)
                
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except PDFProcessingError as e:
            logger.error(f"PDF processing error: {e}")
            raise HTTPException(status_code=422, detail=str(e))
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"Unexpected error processing PDFs: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing PDFs: {str(e)}")
    
    async def chat(self, message: ChatMessage) -> ChatResponse:
        """Enhanced chat with better error handling and context"""
        try:
            if not self.rag_system.is_ready():
                raise HTTPException(
                    status_code=400, 
                    detail="Please upload at least one PDF file to start chatting. Use the /upload endpoint to upload PDFs."
                )
            
            # Validate message length
            if len(message.message.strip()) < 3:
                raise HTTPException(
                    status_code=400,
                    detail="Question must be at least 3 characters long"
                )
            
            if len(message.message) > 1000:
                raise HTTPException(
                    status_code=400,
                    detail="Question is too long. Please keep it under 1000 characters."
                )
            
            response = await self.rag_system.chat(message.message)
            
            # Enhanced response with source details
            return ChatResponse(
                response=response["answer"],
                sources=response.get("sources", [])
            )
            
        except RAGSystemNotReadyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise  # Re-raise HTTP exceptions as-is
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    
    async def search_documents(self, 
                             query: str = Query(..., min_length=3, max_length=200),
                             k: int = Query(default=5, ge=1, le=20)) -> dict:
        """Search documents directly for debugging/inspection"""
        try:
            if not self.rag_system.is_ready():
                raise HTTPException(
                    status_code=400,
                    detail="No documents loaded. Please upload PDFs first."
                )
            
            results = await self.rag_system.search_documents(query, k)
            
            return {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in document search: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    async def get_status(self) -> StatusResponse:
        """Enhanced system status with more details"""
        try:
            ollama_status = await self.rag_system.check_ollama_status()
            system_info = self.rag_system.get_system_info()
            
            return StatusResponse(
                ready=system_info["ready"],
                documents_loaded=system_info["documents_loaded"],
                ollama_status=ollama_status
            )
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            # Return partial status even if there's an error
            return StatusResponse(
                ready=False,
                documents_loaded=0,
                ollama_status=False
            )
    
    async def get_detailed_status(self) -> dict:
        """Get detailed system information"""
        try:
            ollama_status = await self.rag_system.check_ollama_status()
            system_info = self.rag_system.get_system_info()
            
            return {
                **system_info,
                "ollama_status": ollama_status,
                "api_version": "2.0-enhanced"
            }
        except Exception as e:
            logger.error(f"Error getting detailed status: {e}")
            return {
                "ready": False,
                "error": str(e),
                "api_version": "2.0-enhanced"
            }
    
    async def reset_system(self) -> ResetResponse:
        """Reset the RAG system with enhanced logging"""
        try:
            old_doc_count = self.rag_system.document_count  # Fixed method call
            self.rag_system.reset()
            
            logger.info(f"System reset: cleared {old_doc_count} documents")
            
            return ResetResponse(
                message=f"System reset successfully. Cleared {old_doc_count} documents."
            )
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


# Additional Pydantic models for enhanced features
from pydantic import BaseModel
from typing import Dict, Any

class SearchResult(BaseModel):
    content: str
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_found: int

class DetailedStatusResponse(BaseModel):
    ready: bool
    documents_loaded: int
    processed_files: List[str]
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    vector_store_size: int
    ollama_status: bool
    api_version: str