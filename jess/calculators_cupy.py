#!/usr/bin/env python3
"""
The repository for all calculators
"""

import cupy as cp


def to_dtype(data: cp.ndarray, dtype: object) -> cp.ndarray:
    """
    Takes a chunk of data and changes it to a given data type.

    Args:
        data: Array that you want to convert

        dtype: The output data type

    Returns:
        data converted to dtype
    """
    iinfo = cp.iinfo(dtype)

    # Round the data
    cp.around(data, out=data)

    # Clip to stop wrapping
    cp.clip(data, iinfo.min, iinfo.max, out=data)

    return data.astype(dtype)
