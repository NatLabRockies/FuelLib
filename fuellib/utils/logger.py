"""FuelLib logging utilities."""

import logging
from enum import StrEnum


class ANSIColor(StrEnum):
    """ANSI color codes for terminal output."""

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"

    RESET = "\033[0m"


_LEVEL_COLORS = {
    logging.DEBUG: ANSIColor.CYAN,
    logging.INFO: ANSIColor.BLUE,
    logging.WARNING: ANSIColor.YELLOW,
    logging.ERROR: ANSIColor.RED,
    logging.CRITICAL: ANSIColor.BOLD + ANSIColor.RED,
}


class ColorFormatter(logging.Formatter):
    """Formatter that colors the level name of each record."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the record with a colored level name.

        Args:
            record: The log record to format.

        Returns:
            Formatted log record with colored level name.
        """
        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return super().format(record)
        original = record.levelname
        record.levelname = f"{color}{original}{ANSIColor.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


FLLogger = logging.getLogger("fuellib")
FLLogger.setLevel(logging.INFO)

_DEFAULT_FORMAT = "\n%(levelname)s [%(name)s]: %(message)s\n"
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    ColorFormatter(_DEFAULT_FORMAT)
    if _stream_handler.stream.isatty()
    else logging.Formatter(_DEFAULT_FORMAT)
)
FLLogger.addHandler(_stream_handler)


def set_log_level(level: int | str, add_handler: bool = True) -> None:
    """Set the FuelLib logging level.

    Parameters
    ----------
    level : int or str
        Logging level, e.g. ``logging.DEBUG`` or ``"INFO"``.
    add_handler : bool, optional
        If True, ensure the default stream handler (stderr) is attached to the FuelLib
        logger. If False, remove it so the calling application can configure its own
        handlers.
    """
    if isinstance(level, str):
        level = level.upper()
    FLLogger.setLevel(level)

    if add_handler and _stream_handler not in FLLogger.handlers:
        FLLogger.addHandler(_stream_handler)
    elif not add_handler:
        FLLogger.removeHandler(_stream_handler)
