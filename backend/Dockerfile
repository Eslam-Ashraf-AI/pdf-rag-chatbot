# 1. Use an official, lightweight Python base image
FROM python:3.10-slim

# 2. Set MLOps best-practice environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures console output is not buffered by Docker (better logging)
# HF_HOME: Defines a specific cache directory inside the container for HuggingFace weights
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    API_PORT=8000 \
    API_HOST=0.0.0.0

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies required by FAISS and C++ parallel processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy ONLY requirements first to leverage Docker layer caching
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Create a non-root user for security (Huge bonus points for MLOps interviews)
RUN useradd -m mlops_user && \
    mkdir -p /app/.cache/huggingface && \
    chown -R mlops_user:mlops_user /app

# Switch to the non-root user
USER mlops_user

# 9. Expose the port the app runs on
EXPOSE 8000

# 10. Start the FastAPI application directly (Bypassing run.py to decouple Ollama CLI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]