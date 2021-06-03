#!/usr/bin/env python3
"""
The repository for all calculators
"""

import numpy as np
from scipy.stats import entropy


def shannon_entropy(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Calculates the Shannon Entropy along a given axis.

    Return entropy in natural units.

    Args:
        data: 2D Array to calculate entropy

        axis: axis to calculate entropy

    Returns:
        Shannon Entropy along an axis
    """
    if axis == 0:
        data = data.T
    elif axis > 1:
        raise ValueError("Axis out of bounds, given axis=%i" % axis)
    length, _ = data.shape

    entropies = np.zeros(length, dtype=float)
    # Need to loop because np.unique doesn't
    # return counts for all
    for j in range(0, length):
        _, counts = np.unique(
            data[
                j,
            ],
            return_counts=True,
        )
        entropies[j] = entropy(counts)
    return entropies
