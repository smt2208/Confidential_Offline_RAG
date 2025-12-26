"""Vector store management for RAG system."""

from langchain_chroma import Chroma
from backend.src.embeddings import EmbeddingsManager
from backend import config
from backend.logger import get_logger

log = get_logger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store for document embeddings.

    This class handles the persistent storage and retrieval of text embeddings
    using ChromaDB, which stores both the vector representations and associated
    metadata for efficient similarity search.
    """

    def __init__(self, embeddings_manager: EmbeddingsManager,
                 collection_name: str = config.COLLECTION_NAME,
                 persist_directory: str = config.CHROMA_PATH):
        """
        Initialize the vector store manager.

        Args:
            embeddings_manager: Configured embeddings manager for text vectorization
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory path for persistent storage
        """
        self.embeddings_manager = embeddings_manager
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB with persistent storage
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings_manager.get_embeddings(),
            persist_directory=persist_directory
        )

        # Log collection statistics
        try:
            count = self.vectorstore._collection.count()
            log.info(f"VectorStore loaded: {count} chunks in '{collection_name}'")
        except Exception as e:
            log.warning(f"Could not get document count: {e}")

    def get_vectorstore(self) -> Chroma:
        """
        Get the configured vector store instance.

        Returns:
            Chroma: The initialized ChromaDB vector store for document operations
        """
        return self.vectorstore
