#!/usr/bin/env python3
"""
The repository for all calculators
"""

import numpy as np
from scipy.stats import entropy, median_abs_deviation


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


def preprocess(
    data: np.ndarray, central_value_calc: str = "mean", disperion_calc: str = "std"
) -> [np.ndarray, np.ndarray]:
    """
    Preprocess the array for later statistical tests

    Args:
        data: 2D dynamic specra to process

        central_value_calc: The method to calculate the central value,

        dispersion_calc: The method to calculate the dispersion

    Returns:
        data preprocessed along axis=0, axis=1
    """
    central_value_calc = central_value_calc.lower()
    disperion_calc = disperion_calc.lower()
    if central_value_calc == "mean":
        central_0 = np.mean(data, axis=0)
        central_1 = np.mean(data, axis=1)
    elif central_value_calc == "median":
        central_0 = np.median(data, axis=0)
        central_1 = np.median(data, axis=1)
    else:
        raise NotImplementedError(
            f"Given {central_value_calc} for the cental value calculator"
        )

    if disperion_calc == "std":
        dispersion_0 = np.std(data, axis=0, ddof=1)
        dispersion_1 = np.std(data, axis=1, ddof=1)
    if disperion_calc == "mad":
        dispersion_0 = median_abs_deviation(data, axis=0, scale="normal")
        dispersion_1 = median_abs_deviation(data, axis=1, scale="normal")
    else:
        raise NotADirectoryError(f"Given {disperion_calc} for dispersion calculator")

    return (data - central_0) / dispersion_0, (
        data - central_1[:, None]
    ) / dispersion_1[:, None]
