class PDFProcessingError(Exception):
    """Exception raised when PDF processing fails"""
    pass

class OllamaConnectionError(Exception):
    """Exception raised when Ollama connection fails"""
    pass

class RAGSystemNotReadyError(Exception):
    """Exception raised when RAG system is not ready for queries"""
    pass

class InvalidFileTypeError(Exception):
    """Exception raised when non-PDF files are uploaded"""
    pass