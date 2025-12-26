"""Logging configuration for RAG application."""

import logging
import os
from datetime import datetime
from typing import Optional
from backend import config


class LoggerFactory:
    """
    Factory for creating and managing logger instances.

    Uses a singleton-like pattern to ensure logging is configured only once
    and provides cached logger instances to avoid recreation overhead.
    """

    _initialized = False  # Tracks if logging has been configured
    _loggers: dict[str, logging.Logger] = {}  # Cache of created loggers

    @classmethod
    def _setup_logging(cls):
        """
        Setup logging configuration once per application run.

        Configures Python's logging system with:
        - File handler for persistent logs
        - Console handler for real-time output
        - Consistent formatting and log levels
        """
        if cls._initialized:
            return

        log_dir = config.LOG_DIR
        os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists

        # Create daily log file with timestamp
        log_file = os.path.join(log_dir, f"rag_{datetime.now().strftime('%Y%m%d')}.log")

        # Configure logging with both file and console handlers
        logging.basicConfig(
            level=getattr(logging, config.LOG_LEVEL),  # Dynamic log level from config
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),  # Persistent file logging
                logging.StreamHandler()  # Console output for development
            ]
        )

        cls._initialized = True

    @classmethod
    def create(cls, name: str) -> logging.Logger:
        """
        Create or retrieve a logger instance with caching.

        Args:
            name: Logger name (typically __name__ of the calling module)

        Returns:
            logging.Logger: Configured logger instance
        """
        if name not in cls._loggers:
            cls._setup_logging()  # Ensure logging is configured
            cls._loggers[name] = logging.getLogger(name)  # Create and cache logger
        return cls._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """Factory function to get logger instance."""
    return LoggerFactory.create(name)
