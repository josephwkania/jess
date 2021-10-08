#!/usr/bin/env python3

# import logging
import warnings

import cupy as cp
import numpy as np


def _mad_1d_gpu(x, center, nan_policy):
    # Median absolute deviation for 1-d array x.
    # This is a helper function for `median_abs_deviation`; it assumes its
    # arguments have been validated already.  In particular,  x must be a
    # 1-d numpy array, center must be callable, and if nan_policy is not
    # 'propagate', it is assumed to be 'omit', because 'raise' is handled
    # in `median_abs_deviation`.
    # No warning is generated if x is empty or all nan.
    isnan = cp.isnan(x)
    if isnan.any():
        if nan_policy == "propagate":
            return cp.nan
        x = x[~isnan]
    if x.size == 0:
        # MAD of an empty array is nan.
        return cp.nan
    # Edge cases have been handled, so do the basic MAD calculation.
    med = center(x)
    mad = cp.median(cp.abs(x - med))
    return mad


def _contains_nan(a, nan_policy="propagate"):
    policies = ["propagate", "raise", "omit"]
    if nan_policy not in policies:
        raise ValueError(
            "nan_policy must be one of {%s}" % ", ".join("'%s'" % s for s in policies)
        )
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid="ignore"):
            contains_nan = cp.isnan(cp.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = cp.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = "omit"
            warnings.warn(
                "The input array could not be properly checked for nan "
                "values. nan values will be ignored.",
                RuntimeWarning,
            )

    if contains_nan and nan_policy == "raise":
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy


def median_abs_deviation(
    x, axis=0, center=cp.median, scale=1.0, nan_policy="propagate"
):
    r"""
    Compute the median absolute deviation of the data along the given axis.
    The median absolute deviation (MAD, [1]_) computes the median over the
    absolute deviations from the median. It is a measure of dispersion
    similar to the standard deviation but more robust to outliers [2]_.
    The MAD of an empty array is ``np.nan``.
    .. versionadded:: 1.5.0
    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the
        function signature ``func(arr, axis)``.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The default is 1.0. The string "normal" is also accepted,
        and results in `scale` being the inverse of the standard normal
        quantile function at 0.75, which is approximately 0.67449.
        Array-like scale is also allowed, as long as it broadcasts correctly
        to the output such that ``out / scale`` is a valid operation. The
        output dimensions depend on the input array, `x`, and the `axis`
        argument.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    mad : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.
    See Also
    --------
    numpy.std, numpy.var, numpy.median, scipy.stats.iqr, scipy.stats.tmean,
    scipy.stats.tstd, scipy.stats.tvar
    Notes
    -----
    The `center` argument only affects the calculation of the central value
    around which the MAD is calculated. That is, passing in ``center=np.mean``
    will calculate the MAD around the mean - it will not calculate the *mean*
    absolute deviation.
    The input array may contain `inf`, but if `center` returns `inf`, the
    corresponding MAD for that data will be `nan`.
    References
    ----------
    .. [1] "Median absolute deviation",
           https://en.wikipedia.org/wiki/Median_absolute_deviation
    .. [2] "Robust measures of scale",
           https://en.wikipedia.org/wiki/Robust_measures_of_scale
    Examples
    --------
    When comparing the behavior of `median_abs_deviation` with ``np.std``,
    the latter is affected when we change a single value of an array to have an
    outlier value while the MAD hardly changes:
    >>> from scipy import stats
    >>> x = stats.norm.rvs(size=100, scale=1, random_state=123456)
    >>> x.std()
    0.9973906394005013
    >>> stats.median_abs_deviation(x)
    0.82832610097857
    >>> x[0] = 345.6
    >>> x.std()
    34.42304872314415
    >>> stats.median_abs_deviation(x)
    0.8323442311590675
    Axis handling example:
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> stats.median_abs_deviation(x)
    array([3.5, 2.5, 1.5])
    >>> stats.median_abs_deviation(x, axis=None)
    2.0
    Scale normal example:
    >>> x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    >>> stats.median_abs_deviation(x)
    1.3487398527041636
    >>> stats.median_abs_deviation(x, scale='normal')
    1.9996446978061115
    """
    if not callable(center):
        raise TypeError(
            "The argument 'center' must be callable. The given "
            f"value {repr(center)} is not callable."
        )

    # An error may be raised here, so fail-fast, before doing lengthy
    # computations, even though `scale` is not used until later
    if isinstance(scale, str):
        if scale.lower() == "normal":
            scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = cp.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return cp.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return cp.nan
        return cp.full(nan_shape, cp.nan)

    contains_nan, nan_policy = _contains_nan(x, nan_policy)
    x = cp.array(x)
    if contains_nan:
        if axis is None:
            mad = _mad_1d_gpu(x.ravel(), center, nan_policy)
        else:
            mad = cp.apply_along_axis(_mad_1d_gpu, axis, x, center, nan_policy)
    else:
        if axis is None:
            med = center(x, axis=None)
            mad = cp.median(cp.abs(x - med))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            med = cp.expand_dims(center(x, axis=axis), axis)
            mad = cp.median(cp.abs(x - med), axis=axis)

    return mad / scale


def median_abs_deviation_med(
    x: cp.ndarray,
    axis: int = 0,
    center: object = cp.median,
    scale: float = 1.0,
    nan_policy: str = "propagate",
):
    r"""
    Compute the median absolute deviation of the data along the given axis.
    The median absolute deviation (MAD, [1]_) computes the median over the
    absolute deviations from the median. It is a measure of dispersion
    similar to the standard deviation but more robust to outliers [2]_.
    The MAD of an empty array is ``np.nan``.
    .. versionadded:: 1.5.0
    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the range is computed. Default is 0. If None, compute
        the MAD over the entire array.
    center : callable, optional
        A function that will return the central value. The default is to use
        np.median. Any user defined function used will need to have the
        function signature ``func(arr, axis)``.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The default is 1.0. The string "normal" is also accepted,
        and results in `scale` being the inverse of the standard normal
        quantile function at 0.75, which is approximately 0.67449.
        Array-like scale is also allowed, as long as it broadcasts correctly
        to the output such that ``out / scale`` is a valid operation. The
        output dimensions depend on the input array, `x`, and the `axis`
        argument.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    mad : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.

    center : The centers of each array
    See Also
    --------
    numpy.std, numpy.var, numpy.median, scipy.stats.iqr, scipy.stats.tmean,
    scipy.stats.tstd, scipy.stats.tvar
    Notes
    -----
    Modifed from scipy.stats.median_abs_devation

    The `center` argument only affects the calculation of the central value
    around which the MAD is calculated. That is, passing in ``center=np.mean``
    will calculate the MAD around the mean - it will not calculate the *mean*
    absolute deviation.
    The input array may contain `inf`, but if `center` returns `inf`, the
    corresponding MAD for that data will be `nan`.
    References
    ----------
    .. [1] "Median absolute deviation",
           https://en.wikipedia.org/wiki/Median_absolute_deviation
    .. [2] "Robust measures of scale",
           https://en.wikipedia.org/wiki/Robust_measures_of_scale
    Examples
    --------
    When comparing the behavior of `median_abs_deviation` with ``np.std``,
    the latter is affected when we change a single value of an array to have an
    outlier value while the MAD hardly changes:
    >>> from scipy import stats
    >>> x = stats.norm.rvs(size=100, scale=1, random_state=123456)
    >>> x.std()
    0.9973906394005013
    >>> stats.median_abs_deviation(x)
    0.82832610097857
    >>> x[0] = 345.6
    >>> x.std()
    34.42304872314415
    >>> stats.median_abs_deviation(x)
    0.8323442311590675
    Axis handling example:
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> stats.median_abs_deviation(x)
    array([3.5, 2.5, 1.5])
    >>> stats.median_abs_deviation(x, axis=None)
    2.0
    Scale normal example:
    >>> x = stats.norm.rvs(size=1000000, scale=2, random_state=123456)
    >>> stats.median_abs_deviation(x)
    1.3487398527041636
    >>> stats.median_abs_deviation(x, scale='normal')
    1.9996446978061115
    """
    if not callable(center):
        raise TypeError(
            "The argument 'center' must be callable. The given "
            f"value {repr(center)} is not callable."
        )

    # An error may be raised here, so fail-fast, before doing lengthy
    # computations, even though `scale` is not used until later
    if isinstance(scale, str):
        if scale.lower() == "normal":
            scale = 0.6744897501960817  # special.ndtri(0.75)
        else:
            raise ValueError(f"{scale} is not a valid scale value.")

    x = cp.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return cp.nan, cp.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return cp.nan, cp.nan
        return cp.full(nan_shape, cp.nan), cp.nan

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan:
        if axis is None:
            mad = _mad_1d_gpu(x.ravel(), center, nan_policy)
            centers = cp.nanmedian(x.ravel())
        else:
            mad = cp.apply_along_axis(_mad_1d_gpu, axis, x, center, nan_policy)
            centers = cp.nanmedian(x, axis=axis)
    else:
        if axis is None:
            centers = center(x, axis=None)
            mad = cp.median(cp.abs(x - centers))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            centers = center(x, axis=axis)
            med = cp.expand_dims(centers, axis)
            mad = cp.median(cp.abs(x - med), axis=axis)

    return mad / scale, centers
