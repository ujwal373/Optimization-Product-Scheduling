"""
Logging configuration for MAPSO

Provides structured logging using loguru with custom formatting and rotation.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Remove default handler
logger.remove()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for MAPSO

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only logs to console.
        rotation: When to rotate log file (e.g., "100 MB", "1 week")
        retention: How long to keep old log files
        format_string: Custom format string. If None, uses default.
    """
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logging initialized at {level} level")


def get_logger(name: str):
    """
    Get a logger instance with a specific name

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize with default settings on import
setup_logging()
