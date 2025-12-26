"""Configuration settings for Confidential Offline RAG application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file in project root
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
load_dotenv(os.path.join(project_root, ".env"))

# ============================================================================
# API SETTINGS
# ============================================================================
API_HOST = "0.0.0.0"  # Bind to all interfaces
API_PORT = 8000      # FastAPI server port

# ============================================================================
# OLLAMA SETTINGS (External LLM Service)
# ============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama server URL
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")                   # Primary LLM model
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:335m")  # Embedding model

# ============================================================================
# VECTOR STORE SETTINGS (ChromaDB)
# ============================================================================
CHROMA_PATH = os.path.join(backend_dir, "chroma_db")  # Persistent vector database location
COLLECTION_NAME = "records"                            # ChromaDB collection name

# ============================================================================
# INDEXING SETTINGS (Document Processing)
# ============================================================================
CHUNK_SIZE = 400      # Size of text chunks for embedding (in characters)
CHUNK_OVERLAP = 50    # Overlap between consecutive chunks (for context continuity)
TOP_K_PARENT_DOCS = 2 # Number of complete documents to retrieve per query

# ============================================================================
# CHECKPOINT SETTINGS (Conversation Persistence)
# ============================================================================
CHECKPOINT_PATH = os.path.join(backend_dir, "checkpoints", "langgraph_checkpoints.sqlite")  # SQLite for conversation state

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOG_DIR = os.path.join(backend_dir, "logs")  # Log file directory
LOG_LEVEL = "INFO"                           # Logging verbosity (DEBUG, INFO, WARNING, ERROR)

# ============================================================================
# DATA SETTINGS
# ============================================================================
DATA_PATH = os.path.join(project_root, "Data", "Data")  # Input document directory

# ============================================================================
# LLM SETTINGS
# ============================================================================
LLM_TEMPERATURE = 0.2  # Response creativity (0.0 = deterministic, 1.0 = creative)
SEARCH_TYPE = "similarity"  # Vector search method (similarity, mmr, etc.)
