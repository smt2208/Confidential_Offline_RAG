"""Simplified RAG pipeline - main entry point."""

import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.src.embeddings import EmbeddingsManager
from backend.src.vectorstore import VectorStoreManager
from backend.src.retriever import RetrieverManager
from backend.src.llm import LLMManager
from backend.src.workflow import RAGWorkflow
from backend import config
from backend.logger import get_logger

log = get_logger(__name__)


class RAGSystem:
    """
    Main RAG system - simplified orchestrator.

    This class initializes all RAG components and builds the LangGraph workflow
    for processing user queries through the RAG pipeline.
    """

    def __init__(self):
        """Initialize the complete RAG system with all components."""
        log.info("Initializing RAG system...")

        # Initialize core RAG components
        self.embeddings = EmbeddingsManager()  # Handles text embeddings via Ollama
        self.vectorstore = VectorStoreManager(self.embeddings)  # Manages ChromaDB vector storage
        self.retriever = RetrieverManager(self.vectorstore, search_type=config.SEARCH_TYPE)  # Handles document retrieval
        self.llm = LLMManager()  # Manages Ollama LLM for generation

        # Setup conversation persistence using SQLite checkpointer
        checkpoint_dir = os.path.dirname(config.CHECKPOINT_PATH)
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists
        conn = sqlite3.connect(config.CHECKPOINT_PATH, check_same_thread=False)  # Thread-safe SQLite connection
        self.checkpointer = SqliteSaver(conn)  # LangGraph checkpointer for conversation state

        # Build and compile the RAG workflow graph
        self.workflow = RAGWorkflow(self.llm, self.retriever)
        self.graph = self.workflow.build_graph(self.checkpointer)

        log.info("RAG system ready!")

    def get_graph(self):
        """Get the compiled LangGraph workflow for processing queries."""
        return self.graph


def initialize_rag():
    """
    Initialize and return the RAG system graph.

    This is the main entry point for setting up the RAG pipeline.
    Used by the FastAPI application to initialize the system on startup.
    """
    system = RAGSystem()
    return system.get_graph()
