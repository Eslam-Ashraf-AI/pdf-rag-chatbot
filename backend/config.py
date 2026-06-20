import os

class Settings:
    # API Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
        
    # Ollama Settings
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
        
    # OPTIMIZED RAG Settings for better recall and precision
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))  # Balanced for context retention
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "250"))  # Good overlap for context continuity
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "8"))  # Retrieve more chunks for better coverage
        
    # LLM Settings - optimized for consistent responses
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))  # Low temp for consistency
    max_tokens: int = int(os.getenv("MAX_TOKENS", "4096"))
        
    # New RAG Quality Settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))  # Balanced threshold
    min_similarity_score: float = float(os.getenv("MIN_SIMILARITY_SCORE", "0.3"))  # Minimum similarity for retrieval
    max_chunks_per_query: int = int(os.getenv("MAX_CHUNKS_PER_QUERY", "8"))  # Maximum chunks to use per query
        
    # CORS Settings
    cors_origins: list = ["http://localhost:3000", "http://localhost:5173"]

settings = Settings()