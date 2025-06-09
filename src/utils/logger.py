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
def get_logger(name: str = None, log_file: str = "logs/scotus_ai.log", log_level: str = "INFO"):
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module).
        log_file: Path to log file.
        log_level: Logging level.
        
    Returns:
        Logger instance.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return Logger(log_file=log_file, log_level=log_level).get_logger()


# Default logger
scotus_logger = get_logger("scotus_ai") 