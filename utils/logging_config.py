"""Structured logging configuration for AI QA Validator."""

import logging
import sys
from typing import Optional
from contextvars import ContextVar


# Context variable for request ID tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds request ID and structured fields to log records."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured fields.
        
        Adds request_id from context if available.
        """
        # Get request ID from context
        request_id = request_id_var.get()
        if request_id:
            record.request_id = request_id
        else:
            record.request_id = "no-request-id"
        
        # Add question context if available
        if not hasattr(record, 'question_number'):
            record.question_number = ""
        
        return super().format(record)


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create formatter with structured fields
    formatter = StructuredFormatter(
        fmt='%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with structured formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set level for third-party loggers to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def set_request_id(request_id: str) -> None:
    """
    Set the request ID in the current context.
    
    Args:
        request_id: The request ID to set
    """
    request_id_var.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Get the request ID from the current context.
    
    Returns:
        The current request ID or None
    """
    return request_id_var.get()


def log_with_question_context(
    logger: logging.Logger,
    level: int,
    message: str,
    question_number: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log a message with question context.
    
    Args:
        logger: The logger instance
        level: Logging level (e.g., logging.INFO)
        message: Log message
        question_number: Question number for context
        **kwargs: Additional keyword arguments for logging
    """
    extra = kwargs.get('extra', {})
    if question_number:
        extra['question_number'] = question_number
        message = f"[Q:{question_number}] {message}"
    
    kwargs['extra'] = extra
    logger.log(level, message, **kwargs)
