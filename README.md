# JESS - Just in Time Elimination of Spurious Signals

`jess` is Python library to analyze and remove Radio Frequency Interference (RFI) for radio search mode data.

# Overview
Radio Frequency Interference are anthropomorphic signals that can corrupt radio observations. In the context of Fast Radio Burst (FRB) and Pulsar searches, RFI can significantly reduce sensitivity to the astronomical signal and increase the amount of false positives. jess aims to provide a set flexible Python filters that should work on data from a wide variety of telescopes. We use Cupy to optionally leverage Graphical Processing Units (GPUs) to greatly accelerate the filters.

## Command Lines Interfaces Highlights
- `jess_composite.py` A composite MAD/FFT/Highpass 2D filter
- `jess_gauss.py` Use Kurtosis and Skew as a 2D Gaussianity filter
- `rfi_view.py` View 2D dynamic spectra, bandpass, time series, and summery statistics.
- `channel_mask_maker.py` Make channel makes basked on user specified statistics.
- `jess_combine_mocks.py` A clone of the your script that applies the composite filter.

## API
- `channel_masks.py` Make channel masks based on statistics and outlier algorithms
- `dispersion/dispersion_cupy` dispersion routines, roll and FDMT
- `fitters/fitters_cupy` Useful curve fitting, robust spline, arPLS, interactive polynomial, etc
- `JESS_filters/JESS_filters_cupy/JESS_filters_generic` Repository with all the filters
- See the full [API documentation](https://josephwkania.github.io/jess/py-modindex.html)

## Documentation
We have a [docs website](https://josephwkania.github.io/jess/index.html)

# Installation
To install directly into your current Python environment
```bash
pip install git+https://github.com/josephwkania/jess.git
```
If you want a local version
```bash
git clone https://github.com/josephwkania/jess.git
pip install jess
```

If you have a GPU to use, `pip install jess[cupy]`, for tests `pip install jess[tests]`, and for
doc `pip install jess[docs]`

# Questions + Contributing
See [CONTRIBUTING.md](https://github.com/josephwkania/jess/tree/master/CONTRIBUTING.md)
