"""
Utility functions for RAG system.
Includes logging, error handling, token counting, and performance monitoring.
"""

import logging
import time
import functools
from typing import Callable, Any, List, Dict
from pathlib import Path
import tiktoken


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration for the RAG system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("RAG_System")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on error.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger("RAG_System")
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff

            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception

        return wrapper

    return decorator


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger("RAG_System")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result

    return wrapper


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: Input text to count tokens for
        model: Model name to use for tokenization

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_token_limit(
    text: str, max_tokens: int, model: str = "gpt-3.5-turbo"
) -> str:
    """
    Truncate text to fit within a token limit.

    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        model: Model name to use for tokenization

    Returns:
        Truncated text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def validate_data_format(data: Any, expected_type: type) -> bool:
    """
    Validate data format matches expected type.

    Args:
        data: Data to validate
        expected_type: Expected data type

    Returns:
        True if data matches expected type, False otherwise
    """
    return isinstance(data, expected_type)


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def format_context_for_llm(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into context string for LLM.

    Args:
        chunks: List of retrieved chunks with metadata

    Returns:
        Formatted context string
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown")

        context_parts.append(f"[Source {i}: {source}]\n{text}\n")

    return "\n".join(context_parts)


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger("RAG_System")

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.metrics[operation] = {"start": time.time()}

    def end_timer(self, operation: str):
        """End timing an operation and log the duration."""
        if operation in self.metrics and "start" in self.metrics[operation]:
            duration = time.time() - self.metrics[operation]["start"]
            self.metrics[operation]["duration"] = duration
            self.logger.info(f"{operation} completed in {duration:.4f} seconds")
            return duration
        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
