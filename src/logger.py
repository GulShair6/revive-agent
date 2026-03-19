from loguru import logger
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Remove default sink (the one that prints to stderr)
logger.remove()

# Console sink – colorful, good for development
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",  # during dev we want everything
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# File sink – rotating daily + retention
logger.add(
    LOG_DIR / "reviveagent_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # new file at midnight
    retention="10 days",  # keep last 10 days
    level="INFO",  # files get INFO+, console gets DEBUG+
    compression="zip",  # compress old files
    serialize=False,  # human-readable by default
    enqueue=True,  # async write → better performance
)

# Optional: JSON sink for production monitoring (uncomment later)
# logger.add(
#     LOG_DIR / "reviveagent_json_{time:YYYY-MM-DD}.json",
#     format="{time} {level} {message}",
#     level="INFO",
#     serialize=True,
#     rotation="500 MB",
# )

# Export the logger so other modules can just do: from src.logger import logger
__all__ = ["logger"]
