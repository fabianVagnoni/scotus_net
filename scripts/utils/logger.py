"""Logging utilities for SCOTUS AI project."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class Logger:
    """Custom logger class using loguru."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO"):
        """Initialize logger.
        
        Args:
            log_file: Path to log file. If None, logs only to console.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        # Remove default logger
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
        
        # Add file handler if log_file is provided
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="10 MB",
                retention="30 days",
                compression="zip"
            )
        
        self.logger = logger
    
    def get_logger(self):
        """Get the logger instance."""
        return self.logger


# Global logger instance
def get_logger(name: str = None, log_file: str = None, log_level: str = "INFO"):
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module).
        log_file: Path to log file. If None, uses environment variable LOG_FILE or default.
        log_level: Logging level.
        
    Returns:
        Logger instance.
    """
    import os
    
    # Use provided log_file, or environment variable, or default
    if log_file is None:
        log_file = os.environ.get("LOG_FILE", "logs/scotus_ai.log")
    
    # Try to create the log directory and handle permission issues
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test if we can write to the directory
        test_file = log_path.parent / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        return Logger(log_file=log_file, log_level=log_level).get_logger()
        
    except (PermissionError, OSError) as e:
        # Fallback to console-only logging if file logging fails
        print(f"Warning: Cannot create log file {log_file}: {e}")
        print("Falling back to console-only logging")
        return Logger(log_file=None, log_level=log_level).get_logger()


# Default logger
scotus_logger = get_logger("scotus_ai") 