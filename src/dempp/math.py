import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger("pyasp")


"""Math and statistics utilities."""


def compute_nmad(data: np.ndarray) -> float:
    """Calculate the normalized median absolute deviation (NMAD) of the data.

    Args:
        data (np.ndarray): Input data.

    Returns:
        float: NMAD of the data.
    """
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def round_to_decimal(
    x: float, decimal: int = 0, func: Callable = np.round, **kwargs
) -> float:
    """Round a number to a specified number of decimal places.

    Args:
        x (float): The number to round.
        decimal (int, optional): The number of decimal places to round to. Defaults to 0.
        func (Callable, optional): The rounding function to use. Defaults to np.round.
        **kwargs: Additional arguments to pass to the rounding function.

    Returns:
        float: The rounded number.
    """
    multiplier = 10.0**decimal
    return func(x / multiplier, **kwargs) * multiplier
