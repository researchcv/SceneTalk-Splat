"""
Logging System
Provides unified logging functionality
"""

import sys
from loguru import logger
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days"
) -> logger:
    """
    Configure logging system
    
    Args:
        log_file: Log file path, outputs to console only if None
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: Log rotation size
        retention: Log retention time
        
    Returns:
        Configured logger object
    """
    # Remove default handler
    logger.remove()
    
    # Add console output
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        colorize=False
    )
    
    # If log file specified, add file output
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8"
        )
    
    return logger


# Create default logger instance
default_logger = setup_logger()

