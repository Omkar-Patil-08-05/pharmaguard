"""
Common Helper Functions
"""

from pathlib import Path
import logging


def create_directory(path: Path):
    """
    Create directory if it doesn't exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(name: str):

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger

def initialize_project_directories():
    """
    Creates all required project directories.
    """

    from ml.utils.config import (
        REPORTS_DIR,
        PLOTS_DIR,
        NUMERICAL_PLOTS_DIR,
        CATEGORICAL_PLOTS_DIR,
        CORRELATION_PLOTS_DIR,
        MODEL_DIR
    )

    directories = [
        REPORTS_DIR,
        PLOTS_DIR,
        NUMERICAL_PLOTS_DIR,
        CATEGORICAL_PLOTS_DIR,
        CORRELATION_PLOTS_DIR,
        MODEL_DIR
    ]

    for directory in directories:
        create_directory(directory)