from astropy.stats import LombScargle
from functools import partial
from os import path
from scipy.interpolate import interp1d
from scipy.optimize import leastsq
from scipy.signal import savgol_filter
from sklearn import linear_model

import math
import numpy as np
import pandas as pd
import scipy.stats as ss

def process_data(data, star_id_col, period_col, curves_dir, save_curve_files=False):
    """
    Extracts additional features from the stars in the given data file using
    their light curves and existing features.

    Parameters
    ---------
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.
    star_id_col : str
        The name of the column containing the star id.
    period_col : str
        The name of the columns containing the light curve period.
    curves_dir : str
        The directory where the light curves are stored.
    save_curve_files : bool
        If True, then the intermediate light curves are saved to the
        curves_dir.
    """
    columns = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "totvar", "quadvar", "fslope", "lc_rms"
        ,  "lc_flux_asymmetry", "sm_phase_rms", "periodicity", "chi_2", "iqr"
        ,  "roms", "ptpv", "fourier_amplitude", "R_21", "R_31", "f_phase"
        ,  "phi_21", "phi_31", "skewness", "kurtosis", "residual_br_fa_ratio"
        ,  "shapiro_wilk", "slopes_10per", "slopes_90per", "cum_sum"
        ,  "neumann_eta", "crosses", "abv_1std", "bel_1std", "abv_1std_slopes"
        ,  "bel_1std_slopes", "num_obs"
        ]
    data = pd.concat([data, pd.DataFrame(columns=columns)])

    extract_func = partial(extract_with_curve, curves_dir, save_curve_files
            , star_id_col, period_col)
    new_data = data.apply(extract_func, axis=1)

    return new_data

def extract_with_curve(curves_dir, save_curve_files, star_id_col, period_col, data):
    """
    Extracts the features from the given star's data with its light curve.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    save_curve_files : bool
        If True, then the intermediate light curves are saved to the
        curves_dir.
    star_id_col : str
        The name of the column containing the star id.
    period_col : str
        The name of the columns containing the light curve period.
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    star_id = data[star_id_col]
    curve_path = get_curve_path(curves_dir, star_id)

    if path.exists(curve_path):
        curve = get_curve(curve_path)

        return extract_features(data, star_id_col, period_col, curve, curves_dir, save_curve_files)
    else:
        return data

def get_curve_path(curves_dir, star_id):
    """
    Returns the file path of the curve in the given curve file directory for
    the given star id.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    star_id : str
        The id of the given star.

    Returns
    -------
    curve_path : str
        The file path of the curve file for the given star.
    """
    curve_file = "%s.csv" % star_id
    curve_path = path.join(curves_dir, curve_file)

    return curve_path

def get_curve(curve_path):
    """
    Gets the light curve from the file at the specified curve_path.

    Uses a custom csv processing method in order to load in data from files
    faster than `pandas.read_csv`.

    Assumes that the data file follows a csv structure where the columns are
    the time, magnitude, and error in that order.

    Parameters
    ----------
    curve_path : str
        The file path of the curve file for the given star.

    Returns
    -------
    light_curve : numpy.ndarray
        The times, magnitudes, and errors of the light curve.
    """
    with open(curve_path, "r") as f:
        lines = f.read().split("\n")
        parts = [line.split(",")[0:4] for line in lines]

        return np.array(parts[1:-1], dtype="float64")

def extract_features(data, star_id_col, period_col, light_curve, curves_dir, save_curve_files):
    """
    Extracts the features from the given light curve and existing data. Also
    saves the generated smoothed and phase shifted curves.

    Expects the data to have "Numerical_ID", "V_(mag)", "Period_(days)", and
    "Amplitude".

    Expects the light curve to have time as "MJD" and magnitude as "Mag".

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.
    star_id_col : str
        The name of the column containing the star id.
    period_col : str
        The name of the columns containing the light curve period.
    light_curve : numpy.ndarray
        The times, magnitudes, and errors of the light curve.
    curves_dir : str
        The directory that the curve files are stored in.
    save_curve_files : bool
        If True, then the intermediate light curves are saved to the
        curves_dir.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    new_data = data.copy()

    columns = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "totvar", "quadvar", "fslope", "lc_rms"
        ,  "lc_flux_asymmetry", "sm_phase_rms", "periodicity", "chi_2", "iqr"
        ,  "roms", "ptpv", "fourier_amplitude", "R_21", "R_31", "f_phase"
        ,  "phi_21", "phi_31", "skewness", "kurtosis", "residual_br_fa_ratio"
        ,  "shapiro_wilk", "slopes_10per", "slopes_90per", "cum_sum"
        ,  "neumann_eta", "crosses", "abv_1std", "bel_1std", "abv_1std_slopes"
        ,  "bel_1std_slopes", "num_obs"
        ]

    star_id = data[star_id_col]
    period = data[period_col]

    times_dirty = light_curve[:,0]
    magnitudes_dirty = light_curve[:,1]
    errors_dirty = light_curve[:,2]

    times, magnitudes, errors = clean_light_curve(times_dirty, magnitudes_dirty, errors_dirty)

    num_obs = times.shape[0]

    phase_times, phase_magnitudes, phase_errors = phase_fold(times, magnitudes, errors, period)
    phase_slopes = curve_slopes(phase_times, magnitudes)

    sm_phase_times, sm_phase_magnitudes = smooth_curve(phase_times, magnitudes)
    sm_phase_slopes = curve_slopes(sm_phase_times, sm_phase_magnitudes)

    #ls_period = lomb_scargle_periodogram(times, magnitudes, errors)

    lt = linear_trend(times, magnitudes)
    mr = magnitude_ratio(magnitudes)
    ms = maximum_slope(times, magnitudes)
    b1std = beyond_1std(magnitudes)
    rcb = r_cor_bor(magnitudes)
    std = np.std(magnitudes)
    mad = median_absolute_deviation(magnitudes)
    mbrp = median_buffer_range_percentage(magnitudes)
    pa = percent_amplitude(magnitudes)

    totvar = total_variation(sm_phase_magnitudes)
    quadvar = total_variation(sm_phase_magnitudes)
    fslope = maximum_slope(sm_phase_times, sm_phase_magnitudes)

    lc_rms = root_mean_square(magnitudes)
    lc_flux_asymmetry = light_curve_flux_asymmetry(magnitudes, lc_rms)
    sm_phase_rms = root_mean_square(sm_phase_magnitudes)
    periodicity = periodicity_metric(lc_rms, sm_phase_rms)

    chi_2 = chi_2_test(magnitudes, errors)
    iqr = interquartile_range(magnitudes)
    roms = robust_median_statistic(magnitudes, errors)
    ptpv = peak_to_peak_variability(magnitudes, errors)

    fourier_order = 3
    fourier_coef = fourier_decomposition(phase_times, phase_magnitudes, fourier_order)
    fourier_amplitude = fourier_R(fourier_coef, 1)
    R_21 = fourier_R_1(fourier_coef, 2)
    R_31 = fourier_R_1(fourier_coef, 3)
    f_phase = fourier_phi(fourier_coef, 1)
    phi_21 = fourier_phi_1(fourier_coef, 2)
    phi_31 = fourier_phi_1(fourier_coef, 3)

    skewness = ss.skew(magnitudes)[0]
    kurtosis = ss.kurtosis(magnitudes)[0]
    residual_br_fa_ratio = residual_bright_faint_ratio(magnitudes)
    shapiro_wilk = ss.shapiro(magnitudes)[0]
    slopes_10per = np.percentile(phase_slopes[np.logical_not(np.isinf(phase_slopes))], 10)
    slopes_90per = np.percentile(phase_slopes[np.logical_not(np.isinf(phase_slopes))], 90)
    cum_sum = cumulative_sum_range(magnitudes)
    neumann_eta = von_neumann_eta(magnitudes)

    crosses = mean_crosses(sm_phase_magnitudes)
    abv_1std = above_1std(sm_phase_magnitudes)
    bel_1std = beyond_1std(sm_phase_magnitudes) - abv_1std

    abv_1std_slopes, bel_1std_slopes = above_below_1std_slopes(sm_phase_slopes)

    new_data[columns] = [lt, mr, ms, b1std, rcb, std, mad, mbrp
        ,  pa, totvar, quadvar, fslope, lc_rms
        ,  lc_flux_asymmetry, sm_phase_rms, periodicity, chi_2, iqr
        ,  roms, ptpv, fourier_amplitude, R_21, R_31, f_phase, phi_21, phi_31
        ,  skewness, kurtosis, residual_br_fa_ratio, shapiro_wilk
        ,  slopes_10per, slopes_90per, cum_sum, neumann_eta, crosses
        ,  abv_1std, bel_1std, abv_1std_slopes, bel_1std_slopes, num_obs
        ]

    if save_curve_files:
        save_curve(curves_dir, star_id, "phase", phase_times, magnitudes, ["phase", "Mag"])
        save_curve(curves_dir, star_id, "sm_phase", sm_phase_times, sm_phase_magnitudes, ["phase", "Mag"])

    return new_data

def clean_light_curve(times_dirty, magnitudes_dirty, errors_dirty):
    """
    Removes duplicate light curve observations and sorts them in chronological
    order.

    Parameters
    ----------
    times_dirty : numpy.ndarray
        The light curve times.
    magnitudes_dirty : numpy.ndarray
        The light curve magnitudes.
    errors_dirty : numpy.ndarray
        The light curve errors.

    Returns
    -------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    errors : numpy.ndarray
        The light curve errors.
    """
    _, clean_ind = np.unique(times_dirty, return_index=True)

    times = times_dirty[clean_ind]
    magnitudes = magnitudes_dirty[clean_ind]
    errors = errors_dirty[clean_ind]

    num_obs = times.shape[0]
    times = times.reshape(num_obs, 1)
    magnitudes = magnitudes.reshape(num_obs, 1)
    errors = errors.reshape(num_obs, 1)

    return times, magnitudes, errors

def save_curve(curves_dir, star_id, curve_name, times, magnitudes, columns):
    curve = pd.DataFrame(list(zip(times[:,0], magnitudes[:,0])), columns=columns)

    curve_path = get_curve_path(curves_dir, "%s_%s" % (star_id, curve_name))

    curve.to_csv(curve_path, index=False)

def lomb_scargle_periodogram(times, magnitudes, errors):
    ls = LombScargle(times[:,0], magnitudes[:,0], errors[:,0])
    frequency, power = ls.autopower(nyquist_factor=100)

    best_frequency = frequency[np.argmax(power)]
    best_period = 1.0 / best_frequency

    return best_period

def linear_trend(times, magnitudes):
    """
    Returns the slope of a linear line fit to the light curve.

    mag(t) = a * t + b
    lt = a

    (D'Isanto et al., 2015) (2.1.3)

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    slope : numpy.float64
        The slope of the line fit to the light curve.
    """
    model = linear_model.LinearRegression()
    model.fit(times, magnitudes)

    return model.coef_[0][0]

def magnitude_ratio(magnitudes):
    """
    Returns the percent of points above the medium magnitude of the light
    curve. Values range from 0.0 to 1.0.

    mr = P(mag > median_mag)

    (D'Isanto et al., 2015) (2.1.7)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    percent_above_median : numpy.float64
        The percent of points above the median magnitude.
    """
    median = np.median(magnitudes)

    points_above_median = magnitudes[magnitudes > median].size

    return points_above_median / magnitudes.size

def maximum_slope(times, magnitudes):
    """
    Returns the maximum slope between two points in the given light curve.

    ms = max(|(mag_i+1 - mag_i) / (t_i+1 - t_i)|)

    (D'Isanto et al., 2015) (2.1.8)

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    max_slope : numpy.float64
        The maximum slope between two points in the light curve.
    """
    max_slope = None

    mag_diffs = magnitudes[1:] - magnitudes[0:-1]
    time_diffs = times[1:] - times[0:-1]

    slopes = np.divide(mag_diffs, time_diffs)

    max_slope = np.max(slopes[np.logical_not(np.isinf(slopes))])

    return max_slope

def beyond_1std(magnitudes):
    """
    Returns the percent of points in the light curve with a magnitude greater
    than one standard deviation away from the mean. Values range from 0.0 to
    1.0.

    bstd1 = P(|mag - mean_mag| > std)

    (D'Isanto et al., 2015) (2.1.2)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    b1std : numpy.float64
        The percent of magnitudes more than 1 standard deviation from the mean.
    """
    std = np.std(magnitudes)
    mean = np.mean(magnitudes)

    abs_deviations = np.absolute(magnitudes - mean)
    gt_1_std = abs_deviations[abs_deviations > std].size

    return gt_1_std / magnitudes.size

def r_cor_bor(magnitudes):
    """
    Returns the percent of points above 1.5 mag over the median magnitude.

    rcb = P(mag > (mag_median + 1.5))

    (D'Isanto et al., 2015) (2.1.12)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    """
    median_mag = np.median(magnitudes)

    above_1_5_median = magnitudes[magnitudes > (median_mag + 1.5)]

    return above_1_5_median.size / magnitudes.size

def median_absolute_deviation(magnitudes):
    """
    Returns the median absolute deviation from the mean magnitude.

    mad = median_i(|x_i - median_j(x_j)|)

    (D'Isanto et al., 2015) (2.1.5)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    med_abs_deviation : numpy.float64
        The median absolute deviation of the magnitudes.
    """
    median = np.median(magnitudes)
    deviations = magnitudes - median
    absolute_deviations = np.absolute(deviations)

    return np.median(absolute_deviations)

def median_buffer_range_percentage(magnitudes):
    """
    Returns the percent of magnitudes whose deviation from the median is less
    than 10% of the median.

    mbrp = P(|x_i - median_j(x_j)| < 0.1 * median_j(x_j))

    (D'Isanto et al., 2015) (2.1.6)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    med_buf_rng_per : numpy.float64
        The median buffer range percentage of the magnitudes.
    """
    median = np.median(magnitudes)

    deviations = magnitudes - median
    absolute_deviations = np.absolute(deviations)

    p_10_median = 0.1 * median
    within_p_10_median = absolute_deviations[absolute_deviations < p_10_median]

    return within_p_10_median.size / magnitudes.size

def percent_amplitude(magnitudes):
    """
    Returns the greater of the differences between the max and median magnitude
    and the min and median magnitude.

    pa = max(|x_max - median(x)|, |x_min - median(x)|)

    (D'Isanto et al., 2015) (2.1.9)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    per_amplitude : numpy.float64
        The percent amplitude of the magnitudes.
    """
    median = np.median(magnitudes)

    max_diff = np.max(magnitudes) - median
    min_diff = np.min(magnitudes) - median

    return max(max_diff, min_diff)

def total_variation(magnitudes):
    """
    Returns the average of absolute differences between neighboring magnitudes.

    This measure is meant to be used only on evenly sampled curves.

    sum_j(|mag_j+1 - mag_j|) / m

    (Faraway et al., 2014)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    totvar : numpy.float64
        The total variation.
    """
    m = magnitudes.size

    mags_m_plus_1 = magnitudes[1:]
    mags_m = magnitudes[0:-1]

    abs_diffs = np.absolute(mags_m_plus_1 - mags_m)

    return np.sum(abs_diffs) / m

def quadratic_variation(magnitudes):
    """
    Returns the average of squared differences between neighboring magnitudes.

    This measure is meant to be used only on evenly sampled curves.

    sum_j((mag_j+1 - mag_j) ^ 2) / m

    (Faraway et al., 2014)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    quadvar : numpy.float64
        The quadratic variation.
    """
    m = magnitudes.size

    mags_m_plus_1 = magnitudes[1:]
    mags_m = magnitudes[0:-1]

    sq_diffs = np.square(mags_m_plus_1 - mags_m)

    return np.sum(sq_diffs) / m

def phase_fold(times, magnitudes, errors, period):
    """
    Folds the given light curve over its period to express the curve in terms
    of phase rather than time.

    The returned values are sorted in phase order.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    errors : numpy.ndarray
        The light curve errors.
    period : numpy.float64
        The light curve period.

    Returns
    -------
    phase_times : numpy.ndarray
        The light curve times in terms of phase.
    phase_magnitudes : numpy.ndarray
        The light curve magnitudes sorted by phase times.
    phase_errors : numpy.ndarray
        The light curve errors sorted by phase times.
    """
    phase_times_unordered = (times % period) / period

    ordered_ind = np.argsort(phase_times_unordered)

    phase_times = phase_times_unordered[ordered_ind]
    phase_magnitudes = magnitudes[ordered_ind]
    phase_errors = errors[ordered_ind]

    num_obs = phase_times.shape[0]
    phase_times = phase_times.reshape(num_obs, 1)
    phase_magnitudes = phase_magnitudes.reshape(num_obs, 1)
    phase_errors = phase_errors.reshape(num_obs, 1)

    return phase_times, phase_magnitudes, phase_errors

def smooth_curve(times, magnitudes):
    """
    Smoothes the given light curve.

    https://stackoverflow.com/questions/28855928/python-smoothing-data

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    smoothed_times : numpy.ndarray
        The smoothed light curve times.
    smoothed_magnitudes : numpy.ndarray
        Tne smoothed light curve magnitudes.
    """
    x = times[:,0]
    y = magnitudes[:,0]

    smoothed_times = np.linspace(np.min(x), np.max(x), 1000)

    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 101, 3
    smoothed_magnitudes = savgol_filter(itp(smoothed_times), window_size, poly_order)

    smoothed_times = smoothed_times.reshape(smoothed_times.size ,1)
    smoothed_magnitudes = smoothed_magnitudes.reshape(smoothed_magnitudes.size ,1)

    return (smoothed_times, smoothed_magnitudes)

def periodicity_metric(light_curve_rms, sm_phase_rms):
    """
    Returns the periodicity of the given data.

    In this implementation, the sigma value is set to 0.

    Q = ((RMS resid)^2 - sigma^2) / ((RMS raw)^2 - sigma^2)

    (Cody et al., 2014) (6.2)

    Parameters
    ----------
    light_curve_rms : numpy.float64
        The root mean square of the light curve magnitudes.
    sm_phase_rms : numpy.float64
        The root mean square of the smoothed phase folded light curve
        magnitudes.

    Returns
    -------
    periodicity : numpy.float64
        The periodicity of the light curve.
    """
    return (sm_phase_rms ** 2) / (light_curve_rms ** 2)

def light_curve_flux_asymmetry(magnitudes, light_curve_rms):
    """
    Returns the light curve flux asymmetry of the given data.

    M = (< d 10% > -d med) / sigma d

    (Cody et al., 2014) (6.3)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    light_curve_rms : numpy.float64
        The root mean square of the light curve magnitudes.

    Returns
    -------
    lc_flux_asymmetry : numpy.float64
        The light curve flux asymmetry of the light curve.
    """
    median = np.median(magnitudes)

    sorted_magnitudes = np.sort(magnitudes)

    p_10 = int(magnitudes.size / 10)
    p_10_high = magnitudes.size - p_10
    p_10_low = p_10

    top_10_p = sorted_magnitudes[p_10_high:]
    bottom_10_p = sorted_magnitudes[:p_10_low]

    top_bottom_10_p = np.concatenate([top_10_p, bottom_10_p])

    mean_top_bottom = np.mean(top_bottom_10_p)

    return (mean_top_bottom - median) / light_curve_rms

def root_mean_square(xs):
    """
    Returns the root mean square of the given data.

    x_rms = sqrt((1/n) * (x_1 ^ 2 + x_2 ^ 2 + ...))

    Parameters
    ----------
    xs : numpy.ndarray
        The data to take the root mean square of.

    Returns
    -------
    rms : numpy.float64
        The root mean square of the given data.
    """
    squares = xs ** 2
    sum_squares = np.sum(squares)

    rms = math.sqrt((len(xs) ** -1) * sum_squares)
    return rms

def chi_2_test(magnitudes, errors):
    """
    Returns the result of performing a chi^2 test.

    chi^2 = sum((mag - m_bar)^2 / error^2)

    m_bar = sum(mag / error^2) / sum(1 / error^2)

    (Sokolovsky et al., 2016) (1) (2)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    errors : numpy.ndarray
        The light curve observation errors.

    Returns
    -------
    chi_2 : numpy.float64
        The result of the chi^2 test.
    """
    num_obs = magnitudes.shape[0]
    errors_sq = np.square(errors)

    m_bar = np.sum(np.divide(magnitudes, errors_sq)) /\
        np.sum(np.divide(np.ones(num_obs), errors_sq))

    chi_2 = np.sum(np.divide(np.square(magnitudes - m_bar), errors_sq))
    return chi_2

def interquartile_range(magnitudes):
    """
    Returns the interquartile range of the magnitudes.

    https://en.wikipedia.org/wiki/Interquartile_range

    (Sokolovsky et al., 2016)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    iqr : numpy.float64
        The result of the chi^2 test.
    """
    num_obs = magnitudes.shape[0]
    per_25 = int(num_obs / 4.0)

    q1 = magnitudes[per_25]
    q3 = magnitudes[num_obs - per_25]

    return (q3 - q1)[0]

def robust_median_statistic(magnitudes, errors):
    """
    Returns the robust median statistic for the given light curve.

    roms = sum(abs(mag - median(mags)) / error) / (N - 1)

    (Sokolovsky et al., 2016) (7)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    errors : numpy.ndarray
        The light curve observation errors.

    Returns
    -------
    roms : numpy.float64
        The robust median statistic of the magnitudes and errors.
    """
    num_obs = magnitudes.shape[0]
    median = np.median(magnitudes)

    abs_diffs = np.absolute(magnitudes - median)
    s = np.sum(np.divide(abs_diffs, errors))
    roms = s / (num_obs - 1)

    return roms

def peak_to_peak_variability(magnitudes, errors):
    """
    Returns the peak to peak variability of the given magnitudes and errors.

    v = (max(mag - error) - min(mag + error)) / (max(mag - error) + min(mag + error)))

    (Sokolovsky et al., 2016) (9)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    errors : numpy.ndarray
        The light curve observation errors.

    Returns
    -------
    ptpv : numpy.float64
        The peak to peak variability of the magnitudes and errors.
    """
    sums = magnitudes + errors
    differences = magnitudes - errors

    min_sum = np.min(sums)
    max_diff = np.max(differences)

    ptpv = (max_diff - min_sum) / (max_diff + min_sum)
    return ptpv

def fourier_decomposition(times, magnitudes, order):
    """
    Fits the given light curve to a cosine fourier series of the given order
    and returns the fit amplitude and phi weights. The coefficents are
    calculated using a least squares fit.

    The fourier series that is fit is the following:

    n = order
    f(time) = A_0 + sum([A_k * cos(2pi * k * time + phi_k) for k in range(1, n + 1)])

    The fourier coeeficients are returned in a list of the following form:

    [A_0, A_1, phi_1, A_2, phi_2, ...]

    Each of the A coefficients will be positive.

    The number of (time, magnitude) values provided must be greater than or
    equal to the order * 2 + 1. This is a requirement of the least squares
    function used for calculating the coefficients.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    order : int
        The order of the fourier series to fit.

    Returns
    -------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    """
    times = times[:,0]
    magnitudes = magnitudes[:,0]

    num_examples = times.shape[0]
    num_coef = order * 2 + 1

    if num_coef > num_examples:
        raise Exception("Too few examples for the specified order. Number of examples must be at least order * 2 + 1. Required: %d, Actual: %d" % (num_coef, num_examples))

    initial_coef = np.ones(num_coef)

    cost_function = partial(fourier_series_cost, times, magnitudes, order)

    fitted_coef, success = leastsq(cost_function, initial_coef)

    final_coef = correct_coef(fitted_coef, order)

    return final_coef

def correct_coef(coef, order):
    """
    Corrects the amplitudes in the given fourier coefficients so that all of
    them are positive.

    This is done by taking the absolute value of all the negative amplitude
    coefficients and incrementing the corresponding phi weights by pi.

    Parameters
    ----------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    order : int
        The order of the fourier series to fit.

    Returns
    -------
    cor_fourier_coef : numpy.ndarray
        The corrected fit fourier coefficients.
    """
    coef = coef[:]
    for k in range(order):
        i = 2 * k + 1
        if coef[i] < 0.0:
            coef[i] = abs(coef[i])
            coef[i + 1] += math.pi

    return coef

def fourier_series_cost(times, magnitudes, order, coef):
    """
    Returns the error of the fourier series of the given order and coefficients
    in modeling the given light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.
    order : int
        The order of the fourier series to fit.
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.

    Returns
    -------
    error : numpy.float64
        The error of the fourier series in modeling the curve.
    """
    return magnitudes - fourier_series(times, coef, order)

def fourier_series(times, coef, order):
    """
    Returns the magnitude values given by applying the fourier series described
    by the given order and coefficients to the given time values.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    order : int
        The order of the fourier series to fit.
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.

    Returns
    -------
    magnitudes : numpy.ndarray
        The calculated light curve magnitudes.
    """
    cos_vals = [coef[2 * k + 1] * np.cos(2 * np.pi * (k + 1) * times + coef[2 * k + 2])
            for k in range(order)]
    cos_sum = np.sum(cos_vals, axis=0)

    return coef[0] + cos_sum

def fourier_R_1(coef, n):
    """
    Returns the caclulated R_n1 value for the given n using the given fourier
    coefficients.

    For example, giving an n value of 3 would yield the R_31 value of the
    fourier series.

    Parameters
    ----------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    n : int
        The n value for the R_n1.

    Returns
    -------
    r_n1 : numpy.float64
        The calculated R_n1 value.
    """
    r_n = fourier_R(coef, n)
    r_1 = fourier_R(coef, 1)
    return r_n / r_1

def fourier_R(coef, n):
    return coef[2 * (n - 1) + 1]

def fourier_phi_1(coef, n):
    """
    Returns the caclulated phi_n1 value for the given n using the given fourier
    coefficients.

    For example, giving an n value of 3 would yield the phi_31 value of the
    fourier series.

    The phi_n1 value returned is restricted to the range of [0, 2pi).

    Parameters
    ----------
    fourier_coef : numpy.ndarray
        The fit fourier coefficients.
    n : int
        The n value for the phi_n1.

    Returns
    -------
    phi_n1 : numpy.float64
        The calculated phi_n1 value.
    """
    phi_n = fourier_phi(coef, n)
    phi_1 = fourier_phi(coef, 1)
    return (phi_n - n * phi_1) % (2 * math.pi)

def fourier_phi(coef, n):
    return coef[2 * (n - 1) + 2]

def residual_bright_faint_ratio(magnitudes):
    """
    Returns the ratio of the average squared variations from the mean of the
    magnitudes fainter and brighter than the mean magnitude.

    (D.-W. Kim et al, 2015) (1 & 2 & 3)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    ratio : numpy.float64
        The residual bright faint ratio of the given magnitudes.
    """
    mean = np.mean(magnitudes)

    brighter = magnitudes[magnitudes > mean]
    fainter = magnitudes[magnitudes < mean]

    resid_brighter = np.mean(np.square(brighter - mean))
    resid_fainter = np.mean(np.square(fainter - mean))

    ratio = resid_fainter / resid_brighter

    return ratio

def cumulative_sum_range(magnitudes):
    """
    Returns the range of the cumulative sum of the given magnitudes.

    (Kim et al., 2014) (6)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    cum_sum_range : numpy.float64
        The range of the cumulative sum of the magnitudes.
    """
    cum_sums = np.cumsum(magnitudes)

    cum_sum_range = np.max(cum_sums) - np.min(cum_sums)

    return cum_sum_range

def von_neumann_eta(magnitudes):
    """
    Returns the von Neumann eta measure of the degree of trends in the given
    magnitudes.

    (Kim et al., 2014) (5)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    eta : numpy.float64
        The von Neumann eta of the magnitudes.
    """
    diffs = magnitudes[1:] - magnitudes[:-1]

    sum_sq_diffs = np.sum(np.square(diffs))

    std = np.std(magnitudes)

    denom = (magnitudes.size - 1) * std ** 2

    eta = sum_sq_diffs / denom

    return eta

def mean_crosses(magnitudes):
    """
    Returns the number of times that the magnitudes cross over the mean with a
    threshold of 0.1 times the standard deviation.

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    crosses : int
        The number of times that the magnitudes cross over the mean.
    """
    mean = np.mean(magnitudes)
    std = np.std(magnitudes)
    dev_above = mean + std * 0.1
    dev_below = mean - std * 0.1

    above = magnitudes[0] > mean

    mags = magnitudes[:,0]

    relevant_mags = mags[np.where( (mags > dev_above) | (mags < dev_below) )]

    relevant_norm_mags = relevant_mags - mean

    rel_mags_a = relevant_norm_mags[0:-1]
    rel_mags_b = relevant_norm_mags[1:]

    are_crosses = np.multiply(rel_mags_a, rel_mags_b)

    crosses = are_crosses[are_crosses < 0.0].size

    return crosses

def above_1std(magnitudes):
    """
    Returns the percent of points in the light curve with a magnitude greater
    than one standard deviation. Values range from 0.0 to 1.0.

    above_std1 = P(mag - mean_mag > std)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    above_1std : numpy.float64
        The percent of magnitudes above 1 standard deviation from the mean.
    """
    std = np.std(magnitudes)
    mean = np.mean(magnitudes)

    gt_1_std = magnitudes[magnitudes - mean > std].size

    return gt_1_std / magnitudes.size

def curve_slopes(times, magnitudes):
    """
    Returns the slopes between the given (time, magnitude) points.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    slopes : numpy.float64
        The slopes between the given points.
    """
    mag_diffs = magnitudes[1:] - magnitudes[0:-1]
    time_diffs = times[1:] - times[0:-1]

    slopes = np.divide(mag_diffs, time_diffs)

    return slopes

def above_below_1std_slopes(slopes):
    """
    Returns the percent of the slopes that are above and below, respecitvely,
    the mean slope by at least one standard deviation.

    Parameters
    ----------
    slopes : numpy.ndarray
        The slopes between the points in the light curve.

    Returns
    -------
    abv_1std_slopes : numpy.float64
        The percent of slopes above 1 standard deviation above the mean slope.
    bel_1std_slopes : numpy.float64
        The percent of slopes above 1 standard deviation below the mean slope.
    """
    mean = np.mean(slopes)
    std = np.std(slopes)

    above = slopes[slopes > mean + std].size
    below = slopes[slopes < mean - std].size

    return above / slopes.size, below / slopes.size
