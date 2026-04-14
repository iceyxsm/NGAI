"""Reproducibility utilities for NGAI experiments."""

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all RNGs.

    Args:
        seed: The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
