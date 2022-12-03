#!/usr/bin/env python3

"""
Cupy versions of scipy functions.
"""
# import logging
import warnings
from typing import List, Tuple, Union

import numpy as np

try:
    import os

    import cupy as xp

    try:
        if int(os.environ["CUDA_VISIBLE_DEVICES"]) < 0:
            raise RuntimeError
    except KeyError:
        pass
    from cupyx.scipy import ndimage

    BACKEND_GPU = True
except (ModuleNotFoundError, RuntimeError):
    xp = np
    from scipy import ndimage

    BACKEND_GPU = False


def _mad_1d_gpu(x, center, nan_policy):
    # Median absolute deviation for 1-d array x.
    # This is a helper function for `median_abs_deviation`; it assumes its
    # arguments have been validated already.  In particular,  x must be a
    # 1-d numpy array, center must be callable, and if nan_policy is not
    # 'propagate', it is assumed to be 'omit', because 'raise' is handled
    # in `median_abs_deviation`.
    # No warning is generated if x is empty or all nan.
    isnan = xp.isnan(x)
    if isnan.any():
        if nan_policy == "propagate":
            return xp.nan
        x = x[~isnan]
    if x.size == 0:
        # MAD of an empty array is nan.
        return xp.nan
    # Edge cases have been handled, so do the basic MAD calculation.
    med = center(x)
    mad = xp.median(xp.abs(x - med))
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
            contains_nan = xp.isnan(xp.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = xp.nan in set(a.ravel())
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
    x, axis=0, center=xp.median, scale=1.0, nan_policy="propagate"
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

    x = xp.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return xp.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return xp.nan
        return xp.full(nan_shape, xp.nan)

    contains_nan, nan_policy = _contains_nan(x, nan_policy)
    x = xp.array(x)
    if contains_nan:
        if axis is None:
            mad = _mad_1d_gpu(x.ravel(), center, nan_policy)
        else:
            mad = xp.apply_along_axis(_mad_1d_gpu, axis, x, center, nan_policy)
    else:
        if axis is None:
            med = center(x, axis=None)
            mad = xp.median(xp.abs(x - med))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            med = xp.expand_dims(center(x, axis=axis), axis)
            mad = xp.median(xp.abs(x - med), axis=axis)

    return mad / scale


def median_abs_deviation_med(
    x: xp.ndarray,
    axis: int = 0,
    center: object = xp.median,
    scale: Union[float, str] = 1.0,
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

    x = xp.asarray(x)

    # Consistent with `np.var` and `np.std`.
    if not x.size:
        if axis is None:
            return xp.nan, xp.nan
        nan_shape = tuple(item for i, item in enumerate(x.shape) if i != axis)
        if nan_shape == ():
            # Return nan, not array(nan)
            return xp.nan, xp.nan
        return xp.full(nan_shape, xp.nan), xp.nan

    contains_nan, nan_policy = _contains_nan(x, nan_policy)

    if contains_nan:
        if axis is None:
            mad = _mad_1d_gpu(x.ravel(), center, nan_policy)
            centers = center(x.ravel())
        else:
            mad = xp.apply_along_axis(_mad_1d_gpu, axis, x, center, nan_policy)
            centers = center(x, axis=axis)
    else:
        if axis is None:
            centers = center(x, axis=None)
            mad = xp.median(xp.abs(x - centers))
        else:
            # Wrap the call to center() in expand_dims() so it acts like
            # keepdims=True was used.
            centers = center(x, axis=axis)
            med = xp.expand_dims(centers, axis)
            mad = xp.median(xp.abs(x - med), axis=axis)

    return mad / scale, centers


def iqr_med(
    x: xp.ndarray,
    axis: int = None,
    rng: Union[Tuple, List] = (25, 75),
    scale: Union[float, str] = 1.0,
    nan_policy: Union[str, None] = "propagate",
    interpolation: str = "linear",
    keepdims: bool = False,
) -> xp.ndarray:
    r"""
    Compute the interquartile range of the data along the specified axis.
    The interquartile range (IQR) is the difference between the 75th and
    25th percentile of the data. It is a measure of the dispersion
    similar to standard deviation or variance, but is much more robust
    against outliers [2]_.
    The ``rng`` parameter allows this function to compute other
    percentile ranges than the actual IQR. For example, setting
    ``rng=(0, 100)`` is equivalent to `numpy.ptp`.
    The IQR of an empty array is `np.nan`.
    .. versionadded:: 0.18.0
    Parameters
    ----------
    x : array_like
        Input array or object that can be converted to an array.
    axis : int or sequence of int, optional
        Axis along which the range is computed. The default is to
        compute the IQR for the entire array.
    rng : Two-element sequence containing floats in range of [0,100] optional
        Percentiles over which to compute the range. Each must be
        between 0 and 100, inclusive. The default is the true IQR:
        `(25, 75)`. The order of the elements is not important.
    scale : scalar or str, optional
        The numerical value of scale will be divided out of the final
        result. The following string values are recognized:
          * 'raw' : No scaling, just return the raw IQR.
            **Deprecated!**  Use `scale=1` instead.
          * 'normal' : Scale by
            :math:`2 \sqrt{2} erf^{-1}(\frac{1}{2}) \approx 1.349`.
        The default is 1.0. The use of scale='raw' is deprecated.
        Array-like scale is also allowed, as long
        as it broadcasts correctly to the output such that
        ``out / scale`` is a valid operation. The output dimensions
        depend on the input array, `x`, the `axis` argument, and the
        `keepdims` flag.
    nan_policy : {'propagate', 'raise', 'omit', `None`}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
          * `None`: Don't check for nans, uses cp.percentile
    interpolation : {'linear', 'lower', 'higher', 'midpoint',
                     'nearest'}, optional
        Specifies the interpolation method to use when the percentile
        boundaries lie between two data points `i` and `j`.
        The following options are available (default is 'linear'):
          * 'linear': `i + (j - i) * fraction`, where `fraction` is the
            fractional part of the index surrounded by `i` and `j`.
          * 'lower': `i`.
          * 'higher': `j`.
          * 'nearest': `i` or `j` whichever is nearest.
          * 'midpoint': `(i + j) / 2`.
    keepdims : bool, optional
        If this is set to `True`, the reduced axes are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array `x`.
    Returns
    -------
    iqr : scalar or ndarray
        If ``axis=None``, a scalar is returned. If the input contains
        integers or floats of smaller precision than ``np.float64``, then the
        output data-type is ``np.float64``. Otherwise, the output data-type is
        the same as that of the input.
    See Also
    --------
    numpy.std, numpy.var
    Notes
    -----
    From https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html
    modifled to also return the median

    This function is heavily dependent on the version of `numpy` that is
    installed. Versions greater than 1.11.0b3 are highly recommended, as they
    include a number of enhancements and fixes to `numpy.percentile` and
    `numpy.nanpercentile` that affect the operation of this function. The
    following modifications apply:
    Below 1.10.0 : `nan_policy` is poorly defined.
        The default behavior of `numpy.percentile` is used for 'propagate'. This
        is a hybrid of 'omit' and 'propagate' that mostly yields a skewed
        version of 'omit' since NaNs are sorted to the end of the data. A
        warning is raised if there are NaNs in the data.
    Below 1.9.0: `numpy.nanpercentile` does not exist.
        This means that `numpy.percentile` is used regardless of `nan_policy`
        and a warning is issued. See previous item for a description of the
        behavior.
    Below 1.9.0: `keepdims` and `interpolation` are not supported.
        The keywords get ignored with a warning if supplied with non-default
        values. However, multiple axes are still supported.
    References
    ----------
    .. [1] "Interquartile range" https://en.wikipedia.org/wiki/Interquartile_range
    .. [2] "Robust measures of scale"
            https://en.wikipedia.org/wiki/Robust_measures_of_scale
    .. [3] "Quantile" https://en.wikipedia.org/wiki/Quantile
    Examples
    --------
    >>> from scipy.stats import iqr
    >>> x = np.array([[10, 7, 4], [3, 2, 1]])
    >>> x
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> iqr(x)
    4.0
    >>> iqr(x, axis=0)
    array([ 3.5,  2.5,  1.5])
    >>> iqr(x, axis=1)
    array([ 3.,  1.])
    >>> iqr(x, axis=1, keepdims=True)
    array([[ 3.],
           [ 1.]])
    """
    x = xp.asarray(x)

    # This check prevents percentile from raising an error later. Also, it is
    # consistent with `np.var` and `np.std`.
    if not x.size:
        return xp.nan

    # An error may be raised here, so fail-fast, before doing lengthy
    # computations, even though `scale` is not used until later
    _scale_conversions = {"raw": 1.0, "normal": 1.3489795003921636}
    if isinstance(scale, str):
        scale_key = scale.lower()
        if scale_key not in _scale_conversions:
            raise ValueError(f"{scale} not a valid scale for `iqr`")
        if scale_key == "raw":
            warnings.warn(
                "use of scale='raw' is deprecated, use scale=1.0 instead",
                np.VisibleDeprecationWarning,
            )
        scale = _scale_conversions[scale_key]

    # Select the percentile function to use based on nans and policy
    if nan_policy is not None:
        contains_nan, nan_policy = _contains_nan(x, nan_policy)
    else:
        contains_nan = False
        nan_policy = "propagate"

    if contains_nan and nan_policy == "omit":
        # cupy does not have nanpercentile
        # it seems like nans are being omitted using cp.percentile
        # so I think this is ok
        if BACKEND_GPU:
            percentile_func = xp.percentile
        else:
            percentile_func = xp.nanpercentile
    else:
        percentile_func = xp.percentile

    if len(rng) != 2:
        raise TypeError("quantile range must be two element sequence")

    if np.isnan(rng).any():
        raise ValueError("range must not contain NaNs")

    rng = sorted(rng)
    pct = percentile_func(
        x, rng + [50], axis=axis, interpolation=interpolation, keepdims=keepdims
    )
    out = xp.subtract(pct[1], pct[0])

    if scale != 1.0:
        out /= scale

    return out, pct[2]


def _moment(
    array: xp.ndarray, axis: int, test: str, mean: Union[xp.ndarray, None] = None
) -> List:
    """
    Calculate the moments for a given test.

    Args:
        array - Array to consider

        axis - Axis to compute values along

        test - Statistical test: multi m2, m3, m4
               skew: m2, m3, kurtosis: m2, m4

        mean - Mean values, if `None` calculates uses nanmean

    Returns:
        Moments need for a given test along the axis
    """
    if mean is None:
        mean = xp.nanmean(array, axis=axis, keepdims=True)
        mean_func = xp.nanmean
    else:
        mean_func = xp.mean
    a_zero_mean = array - mean

    square = a_zero_mean**2
    if test == "multi":
        moments_list: Tuple = (square, square * a_zero_mean, square * square)
    elif test == "skew":
        moments_list = (square, square * a_zero_mean)
    elif test == "kurtosis":
        moments_list = (square, square * square)
    else:
        raise NotADirectoryError(f"You Requested {test} which is not implemented")

    return [mean_func(moment, axis) for moment in moments_list]


def winsorize(
    array: xp.ndarray, sigma: float, chans_per_fit: int, nan_policy: Union[str, None]
) -> xp.ndarray:
    """
    Winsorize a array by clipping values `sigma` above the fit. The trend
    is flitted using the median fitter. The noise is calculated from the
    difference between the array and fit using IQR.

    Args:
        array - Array to Winsorize, processed in place.

        sigma - Sigma to clip at

        chans_per_fit - Channels per fitting order, see
                        `jess.fitters.median_fitter`

        nan_policy - nan policy, if `None` IQR doesn't check for nans

    Returns:
        winsorized array
    """
    fit = ndimage.median_filter(array, size=chans_per_fit, mode="reflect")
    noise, _ = iqr_med(array - fit, scale="normal", nan_policy=nan_policy)
    top_value = fit + sigma * noise
    mask = array > top_value
    array[mask] = top_value[mask]
    return array


def _chk_asarray(array, axis):
    """
    Handles None axis and turning to
    array
    """
    if axis is None:
        array = xp.ravel(array)
        outaxis = 0
    else:
        array = xp.asarray(array)
        outaxis = axis

    if array.ndim == 0:
        array = xp.atleast_1d(array)

    return array, outaxis


def combined(
    array: xp.ndarray,
    axis: int = 0,
    fisher: bool = True,
    bias: bool = True,
    nan_policy: Union[str, None] = "propagate",
    winsorize_args: Union[Tuple, None] = None,
):
    r"""
    Compute the kurtosis (Fisher or Pearson) and skewness of a dataset.
    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.
    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    For normally distributed data, the skewness should be about zero. For
    unimodal continuous distributions, a skewness value greater than zero means
    that there is more weight in the right tail of the distribution. The
    function `skewtest` can be used to determine if the skewness value
    is close enough to zero, statistically speaking.
    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `a`.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
          * `None`: Don't check for nans, omit
    winsorize_args : If not `None`, this array gets passed to winsorize, see
                     stats.winsorize

    Returns
    -------
    (skewness : ndarray  kurtosis : array)
        The skewness of values along an axis, returning 0 where all values are
        equal. The kurtosis of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.
    Notes
    -----
    The sample skewness is computed as the Fisher-Pearson coefficient
    of skewness, i.e.
    .. math::
        g_1=\frac{m_3}{m_2^{3/2}}
    where
    .. math::
        m_i=\frac{1}{N}\sum_{n=1}^N(x[n]-\bar{x})^i
    is the biased sample :math:`i\texttt{th}` central moment, and
    :math:`\bar{x}` is
    the sample mean.  If ``bias`` is False, the calculations are
    corrected for bias and the value computed is the adjusted
    Fisher-Pearson standardized moment coefficient, i.e.
    .. math::
        G_1=\frac{k_3}{k_2^{3/2}}=
            \frac{\sqrt{N(N-1)}}{N-2}\frac{m_3}{m_2^{3/2}}.
    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.
       Section 2.2.24.1
    Examples
    --------
    >>> from scipy.stats import skew
    >>> combined([1, 2, 3, 4, 5])
    (0.0, 1.7)
    >>> combined([2, 8, 0, 4, 1, 9, 9, 0])
    (0.2650554122698573, 1.333998924716149)
    """
    array, axis = _chk_asarray(array, axis)
    n_el = array.shape[axis]

    if nan_policy is not None:
        _, nan_policy = _contains_nan(array, nan_policy)

    # if contains_nan and nan_policy == "omit":
    #     raise NotImplementedError()
    #     a = ma.masked_invalid(a)
    #     return mstats_basic.skew(a, axis, bias)

    mean = array.mean(axis, keepdims=True)
    # mean = cp.median(a, axis, keepdims=True)
    # m2, m3, m4 = _moment_simple(a, [2, 3, 4], axis, mean=mean)
    m_2, m_3, m_4 = _moment(array, axis, test="multi", mean=mean)

    if winsorize_args is not None:
        m_2 = winsorize(m_2, *winsorize_args, nan_policy=nan_policy)

    with np.errstate(all="ignore"):
        zero = m_2 <= (xp.finfo(m_2.dtype).resolution * mean.squeeze(axis)) ** 2
        vals_skew = xp.where(zero, 0, m_3 / m_2**1.5)
        vals_kurtosis = xp.where(zero, 0, m_4 / m_2**2.0)

    if not bias:
        can_correct_skew = ~zero & (n_el > 2)
        can_correct_kurtosis = ~zero & (n_el > 3)
        if can_correct_skew.any():
            m_2 = xp.extract(can_correct_skew, m_2)
            m_3 = xp.extract(can_correct_skew, m_3)
            nval_skew = xp.sqrt((n_el - 1.0) * n_el) / (n_el - 2.0) * m_3 / m_2**1.5
            xp.place(vals_skew, can_correct_skew, nval_skew)

        if can_correct_kurtosis.any():
            m_2 = xp.extract(can_correct_kurtosis, m_2)
            m_4 = xp.extract(can_correct_kurtosis, m_4)
            nval_kutosis = (
                1.0
                / (n_el - 2)
                / (n_el - 3)
                * ((n_el**2 - 1.0) * m_4 / m_2**2.0 - 3 * (n_el - 1) ** 2.0)
            )
            xp.place(vals_kurtosis, can_correct_kurtosis, nval_kutosis + 3.0)

    if vals_skew.ndim == 0:
        return vals_skew.item(), vals_kurtosis.item()

    if fisher:
        vals_kurtosis -= 3

    return vals_skew, vals_kurtosis
