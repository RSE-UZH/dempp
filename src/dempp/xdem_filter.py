"""Filters to remove outliers and reduce noise in DEMs."""

from __future__ import annotations

import warnings  # noqa: F401

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import numpy as np  # noqa: F401
import scipy  # noqa: F401
from xdem._typing import NDArrayf  # noqa: F401
