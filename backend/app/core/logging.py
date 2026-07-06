import logging
import sys
from backend.app.core.config import settings

def setup_logging():
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Specific logger for the application
    logger = logging.getLogger(settings.PROJECT_NAME)
    logger.setLevel(logging.DEBUG)

    return logger

logger = setup_logging()
