import time
from pathlib import Path
from backend.config.settings import settings
from backend.config.logging import logger

BOOT_TIME: float = time.time()

def create_directories() -> None:
    directories = [
        settings.log_path,
        settings.baseline_path,
        settings.snapshot_path,
        settings.history_path
    ]
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Verified directory: {path.absolute()}")

def on_startup() -> None:
    create_directories()
    banner = f"--- Starting {settings.app_name} v{settings.app_version} ---"
    logger.info(banner)
    logger.info(f"Environment: {'DEBUG' if settings.debug else 'PRODUCTION'}")
    logger.info("Storage directories validated.")
    logger.info("Backend foundation ready.")

def on_shutdown() -> None:
    logger.info(f"--- Shutting down {settings.app_name} ---")
    logger.info("Graceful shutdown complete.")
