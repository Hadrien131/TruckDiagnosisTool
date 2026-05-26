import logging
from typing import Optional


def get_logger(name: str = "truck_diag") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


_logger: Optional[logging.Logger] = None


def get_project_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = get_logger()
    return _logger
