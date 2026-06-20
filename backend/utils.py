import os
import tempfile
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_pdf_files(files) -> List[str]:
    """Validate uploaded files are PDFs and return filenames"""
    invalid_files = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            invalid_files.append(file.filename)
    
    if invalid_files:
        raise ValueError(f"Invalid file types: {', '.join(invalid_files)}. Only PDF files are allowed.")
    
    return [file.filename for file in files]

def create_temp_files(files) -> List[str]:
    """Create temporary files from uploaded files"""
    temp_files = []
    
    for file in files:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_files.append(temp_file.name)
    
    return temp_files

def cleanup_temp_files(temp_files: List[str]):
    """Clean up temporary files"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.debug(f"Deleted temporary file: {temp_file}")
        except OSError as e:
            logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )