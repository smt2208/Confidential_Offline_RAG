"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    """
    Request model for chat API endpoint.

    Contains the user's message and optional thread ID for maintaining
    conversation context across multiple interactions.
    """
    message: str  # The user's chat message
    thread_id: Optional[str] = None  # Optional conversation thread identifier


class ChatResponse(BaseModel):
    """
    Response model for chat API endpoint.

    Contains the AI-generated response and thread ID for conversation continuity.
    """
    response: str  # The AI-generated response text
    thread_id: str  # Conversation thread identifier for follow-up messages


class DocumentInfo(BaseModel):
    """
    Information about a successfully ingested document.

    Used to report the results of document processing during indexing.
    """
    filename: str  # Name of the processed file
    pages: int     # Number of pages in the document
    chunks: int    # Number of text chunks created for embedding


class IngestionResponse(BaseModel):
    """
    Response model for document ingestion operations.

    Reports the success/failure of document processing and provides
    details about each processed document.
    """
    success: bool  # Whether the ingestion operation succeeded
    message: str   # Human-readable status message
    documents: List[DocumentInfo] = []  # List of processed documents (if successful)
