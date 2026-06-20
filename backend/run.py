import subprocess
import sys
import os
import logging

from utils import setup_logging
from config import settings

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def check_ollama():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(
            ["curl", "-s", f"{settings.ollama_base_url}/api/tags"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        return False

def check_model_available():
    """Check if the required model is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return settings.ollama_model in result.stdout
        return False
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False

def main():
    print("🚀 Starting PDF RAG Chatbot Backend...")
    print(f"📋 Configuration:")
    print(f"   • Host: {settings.api_host}:{settings.api_port}")
    print(f"   • Ollama URL: {settings.ollama_base_url}")
    print(f"   • Model: {settings.ollama_model}")
    print(f"   • Embedding Model: {settings.embedding_model}")
    
    # Check if Ollama is running
    if not check_ollama():
        print("❌ Ollama is not running. Please start Ollama first:")
        print("   ollama serve")
        print(f"   ollama pull {settings.ollama_model}")
        sys.exit(1)
    
    print("✅ Ollama is running")
    
    # Check if model is available
    if not check_model_available():
        print(f"⚠️  Model '{settings.ollama_model}' not found. Pulling it now...")
        try:
            subprocess.run(["ollama", "pull", settings.ollama_model], check=True)
            print(f"✅ Model '{settings.ollama_model}' pulled successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to pull model '{settings.ollama_model}'")
            print("   Please run manually: ollama pull {settings.ollama_model}")
            sys.exit(1)
    else:
        print(f"✅ Model '{settings.ollama_model}' is available")
    
    # Start the FastAPI server
    try:
        print(f"🌐 Starting server at http://{settings.api_host}:{settings.api_port}")
        print("📖 API docs available at http://localhost:8000/docs")
        print("Press Ctrl+C to stop...")
        
        subprocess.run([
            "uvicorn", "main:app", 
            "--host", settings.api_host, 
            "--port", str(settings.api_port), 
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()