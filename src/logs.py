"""Terminal logging: coloured and timestamp-free interactively, plain in files."""

import logging
import os
import sys

LOGGER_NAME = "galaxy"

RESET = "\033[0m"
DIM = "\033[2m"
CYAN = "\033[36m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[34m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


def use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


class ColorFormatter(logging.Formatter):
    def __init__(self, color: bool, timestamps: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.color = color
        self.timestamps = timestamps

    def format(self, record: logging.LogRecord) -> str:
        level = f"{record.levelname:<8}"
        tag = record.name
        parts = []

        if self.color:
            level = f"{LEVEL_COLORS.get(record.levelno, '')}{level}{RESET}"
            tag = f"{CYAN}{tag}{RESET}"

        if self.timestamps:
            stamp = self.formatTime(record, self.datefmt)
            parts.append(f"{DIM}{stamp}{RESET}" if self.color else stamp)

        parts.append(f"{level} {tag}  {record.getMessage()}")
        line = "  ".join(parts)

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging(
    name: str = LOGGER_NAME,
    level: int = logging.INFO,
    color: bool | None = None,
) -> logging.Logger:
    """Attach a stdout handler to `name`, once. Returns the logger."""
    logger = logging.getLogger(name)
    color = use_color() if color is None else color

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter(color=color, timestamps=not color))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(level)
    return logger
