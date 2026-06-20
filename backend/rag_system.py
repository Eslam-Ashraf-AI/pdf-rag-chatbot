"""Enhanced RAG system with improved professional response handling and structured output"""
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import re

# Updated LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import settings
from exceptions import PDFProcessingError, OllamaConnectionError, RAGSystemNotReadyError

logger = logging.getLogger(__name__)

class EnhancedPDFRAGSystem:
    """Enhanced PDF RAG System with structured response handling"""
    
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.document_count = 0
        self.processed_documents = []
        self.confidence_threshold = 0.5
        self.original_filenames = {}  # Track original filenames
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with better configuration"""
        try:
            model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            logger.info(f"Embeddings model '{model_name}' initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise PDFProcessingError(f"Failed to initialize embeddings: {e}")
    
    def _initialize_llm(self):
        """Initialize Ollama LLM with better configuration"""
        try:
            self.llm = Ollama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.1,
                num_ctx=getattr(settings, 'max_tokens', 4096),
                top_p=0.9,
                repeat_penalty=1.1
            )
            logger.info(f"Ollama LLM '{settings.ollama_model}' initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise OllamaConnectionError(f"Failed to initialize Ollama: {e}")
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with source preservation"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d+)([A-Z][a-z])', r'\1 \2', text)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _preserve_original_filename(self, file_path: str) -> str:
        """Preserve original filename without modification"""
        # Extract just the filename without the full path
        original_filename = os.path.basename(file_path)
        
        # Store mapping of processed path to original name
        self.original_filenames[file_path] = original_filename
        
        return original_filename
    
    def _create_smart_chunks(self, documents: List[Document]) -> List[Document]:
        """Create smarter chunks with proper source tracking"""
        enhanced_chunks = []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(settings, 'chunk_size', 1000),
            chunk_overlap=getattr(settings, 'chunk_overlap', 200),
            separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
            keep_separator=True
        )
        
        for doc in documents:
            processed_text = self._preprocess_text(doc.page_content)
            doc.page_content = processed_text
            
            # Use original filename
            original_source = doc.metadata.get('source', 'unknown')
            original_filename = self._preserve_original_filename(original_source)
            
            chunks = text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source_file': original_filename,
                    'original_source': original_source,
                    'page_number': chunk.metadata.get('page', 0)
                })
                
                # Add minimal context without cluttering
                context_info = f"Source: {original_filename}"
                if 'page' in chunk.metadata and chunk.metadata['page']:
                    context_info += f" | Page {chunk.metadata['page']}"
                
                chunk.page_content = f"[{context_info}]\n{chunk.page_content}"
                enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _extract_comprehensive_chapter_structure(self, documents: List[Document]) -> Dict[str, Any]:
        """Extract comprehensive chapter structure with improved patterns"""
        structure_info = {
            'chapters': [],
            'sections': [],
            'all_content': []
        }
        
        # Collect all text content for analysis
        full_text = ""
        for doc in documents:
            full_text += f"\n{doc.page_content}"
        
        # Enhanced chapter detection patterns
        chapter_patterns = [
            # Standard chapter formats
            r'Chapter\s+(\d+)[:\s]*([^\n\r]{1,80})',
            r'CHAPTER\s+(\d+)[:\s]*([^\n\r]{1,80})',
            r'Ch\.\s*(\d+)[:\s]*([^\n\r]{1,80})',
            
            # Numbered title formats
            r'^\s*(\d+)\.\s+([A-Z][^\n\r]{5,80})\s*$',
            r'^\s*(\d+)\s+([A-Z][A-Z\s]{10,80})\s*$',
            
            # Table of contents formats
            r'(\d+)\s+([A-Z][^\.\n\r]{10,80})\s+\d+',
            r'Chapter\s+(\d+):\s*([^\n\r]{1,80})',
            
            # Section formats
            r'^\s*(\d+\.\d+)\s+([A-Z][^\n\r]{5,60})\s*$',
            r'^\s*Section\s+(\d+\.\d+)[:\s]*([^\n\r]{1,80})',
        ]
        
        all_matches = []
        
        for pattern in chapter_patterns:
            matches = re.finditer(pattern, full_text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    num, title = match.groups()[:2]
                    
                    # Clean up the title
                    title = re.sub(r'^\s*[:\-\.]?\s*', '', title)
                    title = re.sub(r'\s*\.\s*\d+\s*$', '', title)  # Remove trailing page numbers
                    title = title.strip()
                    
                    if len(title) > 3 and len(title) < 100:  # Reasonable title length
                        match_info = {
                            'number': num.strip(),
                            'title': title,
                            'pattern_used': pattern,
                            'full_match': match.group(0),
                            'start_pos': match.start(),
                            'confidence': self._calculate_match_confidence(pattern, match.group(0), title)
                        }
                        all_matches.append(match_info)
        
        # Sort by confidence and position
        all_matches = sorted(all_matches, key=lambda x: (-x['confidence'], x['start_pos']))
        
        # Deduplicate while preserving best matches
        seen_titles = set()
        seen_numbers = set()
        unique_chapters = []
        
        for match in all_matches:
            title_key = match['title'].lower().strip()
            number_key = match['number']
            
            # Skip if we've seen similar title or exact number
            if title_key not in seen_titles and number_key not in seen_numbers:
                seen_titles.add(title_key)
                seen_numbers.add(number_key)
                unique_chapters.append(match)
        
        # Sort by chapter number
        try:
            unique_chapters = sorted(unique_chapters, key=lambda x: int(x['number']) if x['number'].replace('.', '').isdigit() else float('inf'))
        except:
            unique_chapters = sorted(unique_chapters, key=lambda x: x['number'])
        
        structure_info['chapters'] = unique_chapters
        
        return structure_info
    
    def _calculate_match_confidence(self, pattern: str, full_match: str, title: str) -> float:
        """Calculate confidence score for chapter matches"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for standard patterns
        if 'Chapter' in pattern or 'CHAPTER' in pattern:
            confidence += 0.3
        
        # Check title quality
        if len(title) > 10 and len(title) < 60:
            confidence += 0.1
        
        # Check for proper capitalization
        if title[0].isupper():
            confidence += 0.05
        
        # Check for reasonable content
        if not re.search(r'\d{3,}', title):  # Not just numbers
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    async def process_pdfs(self, pdf_paths: List[str]) -> Dict[str, Any]:
        """Enhanced PDF processing with proper file tracking"""
        try:
            documents = []
            processed_files = []
            
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                    
                logger.info(f"Loading PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    if docs:
                        documents.extend(docs)
                        original_filename = self._preserve_original_filename(pdf_path)
                        processed_files.append(original_filename)
                        logger.info(f"Loaded {len(docs)} pages from {original_filename}")
                    else:
                        logger.warning(f"No content extracted from {pdf_path}")
                        
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_path}: {e}")
                    continue
            
            if not documents:
                raise PDFProcessingError("No valid PDF documents found or processed")
            
            # Extract document structure
            structure_info = self._extract_comprehensive_chapter_structure(documents)
            
            splits = self._create_smart_chunks(documents)
            logger.info(f"Created {len(splits)} document chunks from {len(processed_files)} files")
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    splits, 
                    self.embeddings,
                    distance_strategy="COSINE"
                )
            else:
                new_vector_store = FAISS.from_documents(splits, self.embeddings)
                self.vector_store.merge_from(new_vector_store)
            
            self.processed_documents.extend(processed_files)
            self.document_count += len(processed_files)
            
            self._create_enhanced_qa_chain()
            
            return {
                "chunks_created": len(splits),
                "documents_processed": len(processed_files),
                "files_processed": processed_files,
                "structure_info": structure_info
            }
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Failed to process PDFs: {e}")
    
    def _create_enhanced_qa_chain(self):
        """Create professional QA chain with structured point-based responses"""
        if self.vector_store is None:
            raise RAGSystemNotReadyError("Vector store not initialized")
        
        prompt_template = """You are a professional document assistant. Follow these STRICT guidelines:

CRITICAL RESPONSE FORMAT RULES:
1. ALWAYS structure your response using bullet points (•) or numbered lists (1., 2., 3.)
2. NEVER write in paragraph format - always use structured points
3. Each point should be clear, concise, and focused
4. Use numbered lists for sequential or ordered information
5. Use bullet points (•) for non-sequential items or lists

CONTENT REQUIREMENTS:
1. ONLY use information explicitly stated in the provided context
2. For questions asking for "all" or "complete" information: If the context does not contain the complete answer, start with: "Based on the available content, here are the chapters/sections I can identify:"
3. When listing chapters, sections, or exercises, use this format:
   • Chapter 1: Title Name
   • Chapter 2: Title Name
   OR
   1. Exercise 1.1 - Description
   2. Exercise 1.2 - Description
4. For code examples or functions, format as:
   • Function name: `function_name`
   • Description: Brief description
   • Code structure: Basic structure if available
5. Always include specific page references when available in format: (Page X)
6. If you cannot find specific information, respond with: "• I cannot find this specific information in the uploaded document(s)."

CHAPTER TITLE HANDLING:
- Look specifically for patterns like "Chapter X:", "CHAPTER X", "X. Title", or table of contents entries
- Include both chapter numbers and full titles when available
- If only partial information is available, clearly state what was found

Context from documents:
{context}

Question: {question}

Structured Response (MUST use bullet points or numbered lists - NO paragraphs):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": getattr(settings, 'retrieval_k', 10),  # Increased for better chapter detection
                "fetch_k": 25,
            }
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PROMPT,
                "document_separator": "\n---\n"
            },
            return_source_documents=True,
            verbose=False
        )
        
        logger.info("Enhanced structured QA chain created successfully")
    
    def _format_structured_answer(self, answer: str, source_details: List[Dict]) -> str:
        """Ensure answer is properly structured with bullet points"""
        # Clean up the answer
        answer = re.sub(r'\[Source:.*?\]\s*', '', answer)
        answer = answer.strip()
        
        # Check if answer already has proper structure
        has_bullets = bool(re.search(r'^\s*[•\-\*]', answer, re.MULTILINE))
        has_numbers = bool(re.search(r'^\s*\d+\.', answer, re.MULTILINE))
        
        if not (has_bullets or has_numbers):
            # Convert to bullet points if not already structured
            lines = answer.split('\n')
            structured_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('•') and not line.startswith('-') and not re.match(r'^\d+\.', line):
                    # Add bullet point if it's a substantial line
                    if len(line) > 5:
                        structured_lines.append(f"• {line}")
                    else:
                        structured_lines.append(line)
                else:
                    structured_lines.append(line)
            
            answer = '\n'.join(structured_lines)
        
        # Add source references
        if source_details:
            answer += self._format_source_references(source_details)
        
        return answer
    
    def _format_source_references(self, source_details: List[Dict]) -> str:
        """Format source references professionally"""
        unique_sources = {}
        for detail in source_details:
            file_key = detail["file"]
            if file_key not in unique_sources:
                unique_sources[file_key] = []
            if detail["page"] != "Unknown" and detail["page"] not in unique_sources[file_key]:
                unique_sources[file_key].append(detail["page"])
        
        source_text = "\n\n**Sources:**"
        for file, pages in unique_sources.items():
            if pages and pages != ["Unknown"]:
                page_list = ", ".join(map(str, sorted(set(pages))))
                source_text += f"\n• {file} (Pages: {page_list})"
            else:
                source_text += f"\n• {file}"
        
        return source_text
    
    async def analyze_chapter_structure(self, query: str = "chapters sections table of contents") -> Dict[str, Any]:
        """Enhanced chapter structure analysis"""
        if not self.vector_store:
            return {"error": "No documents loaded"}
        
        # Search for structural information with multiple queries
        structure_queries = [
            "table of contents chapters sections",
            "chapter titles headings structure",
            "chapter 1 chapter 2 chapter 3",
            "contents overview outline"
        ]
        
        all_structure_docs = []
        for sq in structure_queries:
            docs = self.vector_store.similarity_search_with_score(sq, k=8)
            all_structure_docs.extend(docs)
        
        # Remove duplicates and sort by relevance
        unique_docs = {}
        for doc, score in all_structure_docs:
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            if doc_key not in unique_docs or score < unique_docs[doc_key][1]:
                unique_docs[doc_key] = (doc, score)
        
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x[1])[:15]
        
        chapters = []
        for doc, score in sorted_docs:
            text = doc.page_content
            page = doc.metadata.get('page', 0)
            
            # Enhanced chapter detection patterns
            chapter_patterns = [
                r'Chapter\s+(\d+)[:\s]*([^\n\r]{1,80})',
                r'CHAPTER\s+(\d+)[:\s]*([^\n\r]{1,80})',
                r'^(\d+)\.\s+([A-Z][^\n\r]{5,80})\s*$',
                r'^\s*(\d+)\s+([A-Z][A-Z\s]{10,80})\s*$',
                r'(\d+)\s+([A-Z][^\.\n\r]{10,80})\s+\d+',  # TOC format
            ]
            
            for pattern in chapter_patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        num, title = match.groups()[:2]
                        
                        # Clean title
                        title = re.sub(r'^\s*[:\-\.]?\s*', '', title)
                        title = re.sub(r'\s*\.\s*\d+\s*$', '', title)
                        title = title.strip()
                        
                        if len(title) > 3 and len(title) < 100:
                            chapters.append({
                                'number': num.strip(),
                                'title': title,
                                'page': page,
                                'confidence': 1.0 - score,
                                'pattern': pattern
                            })
        
        # Deduplicate and sort
        unique_chapters = {}
        for ch in chapters:
            key = ch['number']
            if key not in unique_chapters or ch['confidence'] > unique_chapters[key]['confidence']:
                unique_chapters[key] = ch
        
        sorted_chapters = sorted(
            unique_chapters.values(), 
            key=lambda x: int(x['number']) if x['number'].isdigit() else 999
        )
        
        return {
            'chapters': sorted_chapters,
            'total_chapters_found': len(sorted_chapters),
            'confidence': 'High' if len(sorted_chapters) >= 3 else 'Medium' if len(sorted_chapters) >= 1 else 'Low'
        }
    
    async def chat(self, question: str) -> Dict[str, Any]:
        """Enhanced chat with structured response handling"""
        try:
            if not self.is_ready():
                raise RAGSystemNotReadyError("System not ready - please upload PDFs first")
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Special handling for chapter/structure questions
            if any(phrase in question.lower() for phrase in [
                'chapter titles', 'all chapters', 'chapter names', 'table of contents',
                'all the chapters', 'list chapters', 'chapters in'
            ]):
                structure_info = await self.analyze_chapter_structure()
                
                if structure_info.get('chapters') and len(structure_info['chapters']) > 0:
                    chapters = structure_info['chapters']
                    
                    # Format as structured list
                    formatted_response = "Based on the available content, here are the chapters I can identify:\n\n"
                    
                    for ch in chapters:
                        if ch['title']:
                            formatted_response += f"• Chapter {ch['number']}: {ch['title']}\n"
                        else:
                            formatted_response += f"• Chapter {ch['number']}\n"
                    
                    if len(chapters) < 5:
                        formatted_response += f"\n• Note: This represents {len(chapters)} chapter(s) found in the document content. There may be additional chapters not captured in the extracted text."
                    
                    sources = list(set([self.processed_documents[0]] if self.processed_documents else ["uploaded_document.pdf"]))
                    pages = [ch['page'] for ch in chapters if ch.get('page', 0) > 0]
                    
                    return {
                        "answer": formatted_response,
                        "sources": sources,
                        "source_details": [{"file": sources[0], "page": p} for p in pages] if pages else [],
                        "question": question,
                        "confidence_score": 0.8,
                        "is_reliable": True
                    }
            
            # Regular processing for other questions
            result = self.qa_chain.invoke({"query": question})
            raw_answer = result["result"].strip()
            
            # Extract source information with original filenames
            sources = []
            source_details = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, "metadata"):
                        original_filename = doc.metadata.get("source_file", "Unknown Document")
                        source_info = {
                            "file": original_filename,
                            "page": doc.metadata.get("page", "Unknown"),
                            "relevance_score": getattr(doc, '_similarity_score', 0.0)
                        }
                        source_details.append(source_info)
                        
                        if original_filename not in sources:
                            sources.append(original_filename)
            
            # Assess answer quality
            quality_assessment = self._assess_answer_quality(question, raw_answer, source_details)
            
            # Format structured response
            if quality_assessment['is_reliable']:
                formatted_answer = self._format_structured_answer(raw_answer, source_details)
            else:
                formatted_answer = "• I cannot find this specific information in the uploaded document(s).\n• Please verify that your question relates to content that is explicitly covered in the documents."
                sources = []
                source_details = []
            
            return {
                "answer": formatted_answer,
                "sources": sources,
                "source_details": source_details,
                "question": question,
                "confidence_score": quality_assessment['confidence_score'],
                "is_reliable": quality_assessment['is_reliable']
            }
            
        except RAGSystemNotReadyError:
            raise
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return {
                "answer": "• I encountered an error while processing your question.\n• Please try again or rephrase your question.",
                "sources": [],
                "source_details": [],
                "question": question,
                "confidence_score": 0.0,
                "is_reliable": False
            }
    
    def _assess_answer_quality(self, question: str, answer: str, sources: List[Dict]) -> Dict[str, Any]:
        """Assess the quality and confidence of the answer"""
        quality_indicators = {
            'has_specific_info': False,
            'has_sources': len(sources) > 0,
            'answer_length_appropriate': 10 < len(answer) < 3000,
            'contains_uncertainty_phrases': any(phrase in answer.lower() for phrase in [
                'not available', 'cannot find', 'not mentioned', 'unclear'
            ]),
            'contains_confident_phrases': any(phrase in answer.lower() for phrase in [
                'based on the document', 'according to the text', 'the document states',
                'specifically mentions', 'clearly indicates', 'explicitly states',
                'found in', 'shown in', 'described in', 'mentioned in'
            ]),
            'is_complete_answer': True,
            'has_concrete_content': False
        }
        
        # Check for specific information patterns
        if re.search(r'\b(chapter|section|page|\d+\.\d+|title|heading|function|example|code|implementation)\b', answer.lower()):
            quality_indicators['has_specific_info'] = True
        
        # Check for concrete content
        concrete_patterns = [
            r'function\s+\w+\s*\(',
            r'\w+\s*=\s*',
            r'=>',
            r'\{[\s\S]*\}',
            r'\d+\.',
            r'example\s*:',
            r'for\s+instance',
            r'such\s+as'
        ]
        
        if any(re.search(pattern, answer, re.IGNORECASE) for pattern in concrete_patterns):
            quality_indicators['has_concrete_content'] = True
        
        # Calculate confidence score
        confidence_score = 0.0
        if quality_indicators['has_sources']:
            confidence_score += 0.2
        if quality_indicators['has_specific_info']:
            confidence_score += 0.2
        if quality_indicators['has_concrete_content']:
            confidence_score += 0.3
        if quality_indicators['contains_confident_phrases']:
            confidence_score += 0.1
        if not quality_indicators['contains_uncertainty_phrases']:
            confidence_score += 0.1
        if quality_indicators['answer_length_appropriate']:
            confidence_score += 0.05
        if quality_indicators['is_complete_answer']:
            confidence_score += 0.05
        
        return {
            'confidence_score': confidence_score,
            'indicators': quality_indicators,
            'is_reliable': confidence_score >= self.confidence_threshold,
        }
    
    # ... (rest of the methods remain the same as in original)
    
    async def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Direct document search for inspection"""
        if not self.vector_store:
            return []
        
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "similarity_score": float(score),
                "metadata": dict(doc.metadata),
                "confidence": "High" if score < 0.5 else "Medium" if score < 0.8 else "Low"
            })
        
        return results
    
    def is_ready(self) -> bool:
        """Check if the system is ready for chat"""
        return (
            self.vector_store is not None and 
            self.qa_chain is not None and 
            self.document_count > 0
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "ready": self.is_ready(),
            "documents_loaded": self.document_count,
            "processed_files": self.processed_documents,
            "confidence_threshold": self.confidence_threshold,
            "embedding_model": getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2'),
            "llm_model": settings.ollama_model,
            "vector_store_size": len(self.vector_store.index_to_docstore_id) if self.vector_store else 0
        }
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for responses"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold set to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    async def check_ollama_status(self) -> bool:
        """Check Ollama status"""
        try:
            test_response = self.llm.invoke("Respond with 'OK' only.")
            return test_response and "OK" in test_response.strip()
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def reset(self):
        """Reset the RAG system"""
        self.vector_store = None
        self.qa_chain = None
        self.document_count = 0
        self.processed_documents = []
        self.original_filenames = {}
        logger.info("RAG system reset successfully")