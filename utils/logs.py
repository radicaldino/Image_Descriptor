import logging
import sys
from .constants import DEBUG

def setup_logger(name: str = "rag_as_a_service", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a console logger only if DEBUG is True"""
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    if not DEBUG:
        # Disable logging when DEBUG is False
        logger.disabled = True
        return logger
    
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
logger = setup_logger()
