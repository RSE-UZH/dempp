import logging
from pathlib import Path

# Import submodules
from . import (
    dod,  # noqa
    elevation_bands,  # noqa
    # filter,  # noqa
    io,  # noqa
    statistics,  # noqa
    utils,  # noqa
)

__version__ = "0.0.1"


def setup_logger(
    level: str | int = logging.INFO,
    name="dempp",
    log_to_file: bool = False,
    log_folder: Path = "./.logs",
):
    """
    Reconfigures the 'dempp' logger with new parameters by calling setup_logger.

    Args:
        level (str | int): The logging level (e.g., 'info', 'debug', 'warning').
        name (str): The name of the logger.
        log_to_file (bool, optional): Whether to log to a file. Defaults to True.
        log_folder (Path, optional): Path to the directory for the log file if log_to_file is True. Defaults to "./.logs".

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
    )


# Setup logger and timer for the package
logger = setup_logger(level=logging.INFO, name="dempp")
timer = utils.Timer(logger=logger)
