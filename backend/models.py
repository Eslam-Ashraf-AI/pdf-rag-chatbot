"""Pydantic models for API requests and responses"""
from pydantic import BaseModel
from typing import List, Optional

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []

class UploadResponse(BaseModel):
    message: str
    files_processed: List[str]
    chunks_created: int

class StatusResponse(BaseModel):
    ready: bool
    documents_loaded: int
    ollama_status: bool

class ResetResponse(BaseModel):
    message: str