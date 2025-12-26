"""LLM management for RAG system."""

from langchain_ollama import ChatOllama
from backend import config
from backend.logger import get_logger

log = get_logger(__name__)


class LLMManager:
    """
    Manages Ollama chat models for text generation.

    This class handles the configuration and initialization of Ollama's chat models,
    which are used for generating responses in the RAG pipeline.
    """

    def __init__(self, model: str = config.OLLAMA_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL,
                 temperature: float = config.LLM_TEMPERATURE):
        """
        Initialize the LLM manager.

        Args:
            model: Name of the Ollama chat model to use
            base_url: URL of the Ollama server
            temperature: Response creativity (0.0 = deterministic, 1.0 = creative)
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        # Initialize Ollama chat model
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature
        )

        log.info(f"LLM initialized: {model}")

    def get_llm(self) -> ChatOllama:
        """
        Get the configured LLM instance.

        Returns:
            ChatOllama: The initialized chat model for text generation
        """
        return self.llm
