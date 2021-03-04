import numpy as np
from  scipy import stats

def bandpass_fiter(channels, bandpass, poly_order=20, mask_sigma=2):

    fit_values = np.polyfit(channels, bandpass, poly_order) # fit a polynomial
    poly = np.poly1d(fit_values) # get the values of the fitted bandpass
    diff = band - poly(channels) # find the differnece betweeen fitted and real bandpass
    std_diff = stats.median_abs_deviation(diff, scale='normal') 
    mask = np.abs(diff-np.median(diff))<mask_sigma*std_diff 

    fit_values_clean = np.polyfit(x[mask], band[mask], 20) # refit masking the outliers
    poly_clean = np.poly1d(fit_values_clean)

    poly_cleaned =  np.poly1d(fit_values)

    return poly_cleaned(channels)
