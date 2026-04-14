"""Tests for ngai.utils."""

import random

import numpy as np

from ngai.utils.seed import set_seed


def test_set_seed_reproducibility() -> None:
    """Verify that set_seed produces deterministic random sequences."""
    set_seed(123)
    a = random.random()
    b = np.random.random()

    set_seed(123)
    assert random.random() == a
    assert np.random.random() == b
