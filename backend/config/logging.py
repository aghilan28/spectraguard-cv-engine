import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from backend.config.settings import settings

def setup_logger() -> logging.Logger:
    logger = logging.getLogger(settings.app_name)
    
    if logger.hasHandlers():
        return logger

    log_level = logging.DEBUG if settings.debug else getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(settings.log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "spectraguard.log"

    file_handler = TimedRotatingFileHandler(
        filename=str(log_file),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
