"""Document retriever for RAG system."""

import os
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.src.vectorstore import VectorStoreManager
from backend import config
from backend.logger import get_logger
from typing import Optional

log = get_logger(__name__)


class RetrieverManager:
    """
    Manages ParentDocumentRetriever for complete file retrieval.

    This class configures a ParentDocumentRetriever that:
    1. Stores complete parent documents (full PDFs) in a key-value store
    2. Creates searchable child chunks from parent documents
    3. Retrieves complete parent documents based on child chunk similarity
    """

    def __init__(self, vectorstore_manager: VectorStoreManager,
                 parent_store_dir: Optional[str] = None,
                 chunk_size: int = config.CHUNK_SIZE,
                 chunk_overlap: int = config.CHUNK_OVERLAP,
                 search_type: str = "similarity"):

        # Setup parent document store directory (defaults to ../parent_docs)
        if parent_store_dir is None:
            parent_store_dir = os.path.join(os.path.dirname(config.CHROMA_PATH), "parent_docs")
        os.makedirs(parent_store_dir, exist_ok=True)

        # Create document store for parent documents using local file storage
        parent_store = LocalFileStore(parent_store_dir)
        docstore = create_kv_docstore(parent_store)

        # Create text splitter for child chunks (searchable units)
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Retrieve more child chunks than final parent docs for better selection
        # This ensures we have good candidates before selecting top-k parents
        vectorstore_k = config.TOP_K_PARENT_DOCS * 2  # Retrieve 2x more child chunks

        search_kwargs = {"k": vectorstore_k}

        # Create the ParentDocumentRetriever
        # - vectorstore: stores child chunks with embeddings
        # - docstore: stores complete parent documents
        # - child_splitter: splits parents into searchable chunks
        # - parent_splitter: None (we store full documents as parents)
        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore_manager.get_vectorstore(),
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=None,  # Store full documents as parents
            search_kwargs=search_kwargs
        )

        self.search_type = search_type
        log.info(f"Retriever configured (parent_docs: {parent_store_dir}, search_type: similarity)")

    
    def retrieve(self, query: str, k: int = 5) -> list:
        """
        Retrieve top k relevant parent documents for a query.

        The ParentDocumentRetriever works by:
        1. Finding the most similar child chunks in the vector store
        2. Returning the complete parent documents that contain those chunks
        3. Ensuring users get full context, not just snippets

        Args:
            query: Search query string
            k: Number of parent documents to return (default: 5, but overridden by config)

        Returns:
            list: List of Document objects (complete parent documents)
        """
        try:
            # Get results from ParentDocumentRetriever
            results = self.retriever.invoke(query)
            # Limit to k results and handle empty results
            retrieved = results[:k] if results else []
            log.info(f"Retrieved {len(retrieved)} documents for query: '{query[:50]}...'")
            return retrieved
        except Exception as e:
            log.error(f"Retrieval error for query '{query[:50]}...': {e}")
            return []

    def get_retriever(self):
        """Get the underlying retriever for advanced usage."""
        return self.retriever

