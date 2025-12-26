"""Embeddings management for RAG system."""

from langchain_ollama import OllamaEmbeddings
from backend import config
from backend.logger import get_logger

log = get_logger(__name__)


class EmbeddingsManager:
    """
    Manages Ollama embeddings for text vectorization.

    This class handles the creation and configuration of Ollama's embedding models,
    which convert text into numerical vectors for semantic search and similarity matching.
    """

    def __init__(self, model: str = config.OLLAMA_EMBEDDING_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL):
        """
        Initialize the embeddings manager.

        Args:
            model: Name of the Ollama embedding model to use
            base_url: URL of the Ollama server
        """
        self.model = model
        self.base_url = base_url
        self.embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        log.info(f"Embeddings initialized: {model}")

    def get_embeddings(self) -> OllamaEmbeddings:
        """
        Get the configured embeddings instance.

        Returns:
            OllamaEmbeddings: The initialized embeddings object for text vectorization
        """
        return self.embeddings
