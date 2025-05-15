import logging
from pathlib import Path

# Import submodules
from . import (
    # dod,
    # elevation_bands,
    # filter,
    io,
    # statistics,
    uncertainty,
    utils,
)

__version__ = "0.0.1"

__all__ = [
    "dod",
    "elevation_bands",
    # "filter",
    "io",
    "statistics",
    "uncertainty",
    "utils",
    "setup_logger",
    "logger",
    "timer",
    "__version__",
]


def setup_logger(
    level: str | int = logging.INFO,
    name="dempp",
    log_to_file: bool = False,
    log_folder: Path = Path("./.logs"),
    redirect_to_stdout: bool = True,
    force: bool = True,
    **kwargs,
) -> logging.Logger:
    """
    Reconfigures the 'dempp' logger with new parameters by calling setup_logger.

    Args:
        level (str | int): The logging level (e.g., 'info', 'debug', 'warning').
        name (str): The name of the logger.
        log_to_file (bool, optional): Whether to log to a file. Defaults to True.
        log_folder (Path, optional): Path to the directory for the log file if log_to_file is True. Defaults to "./.logs".
        redirect_to_stdout (bool, optional): Whether to redirect console output to stdout. Defaults to True.
        force (bool, optional): If True, any existing handlers attached to the root logger are removed and closed before reconfiguration. Defaults to True.

    Returns:
        logging.Logger: The reconfigured 'dempp' logger.

    Example:
        >>> import logging
        >>> from dempp import setup_logger
        >>> logger = setup_logger(level=logging.DEBUG, log_to_file=False)
        >>> logger.debug("This is a debug message")
    """
    return utils.logger.setup_logger(
        level=level,
        name=name,
        log_to_file=log_to_file,
        log_folder=log_folder,
        redirect_to_stdout=redirect_to_stdout,
        force=force,
        **kwargs,
    )


# Setup logger and timer for the package
logger = setup_logger(level=logging.INFO, name="dempp")
timer = utils.Timer(logger=logger)
