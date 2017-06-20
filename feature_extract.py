from functools import partial
from os import path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn import linear_model

import math
import numpy as np
import pandas as pd
import sys

def main():
    """
    $ python3 feature_extract.py CatalinaVars.csv curves/ test.csv
    """
    data_file = sys.argv[1]
    curves_dir = sys.argv[2]
    output_file = sys.argv[3]

    result = validate_arguments(data_file, curves_dir, output_file)

    if isinstance(result, str):
        print(result, file=sys.stderr)
        sys.exit(1)

    main_functionality(data_file, curves_dir, output_file)

def validate_arguments(data_file, curves_dir, output_file):
    """
    Checks to see if the given command line arguments are valid. Returns None
    if they are all valid, or an error string if one or more are invalid.

    Parameters
    ---------
    data_file : str
        The file path of the input data file.
    curves_dir : str
        The directory where the light curves are stored.
    output_file : str
        The file path of where to write the output data file.

    Returns
    -------
    error : Union[None, str]
        The error message if at least one argument was invalid, otherwise is
        None.
    """
    if not path.exists(data_file):
        return "The given data file does not exist: %s" % data_file
    if not path.isfile(data_file):
        return "The given data file is not a file: %s" % data_file

    if not path.exists(curves_dir):
        return "The given curve file directory does not exist: %s" % curves_dir
    if not path.isdir(curves_dir):
        return "The given curve file directory is not a directory: %s" % curves_dir

    return None

def main_functionality(data_file, curves_dir, output_file):
    """
    Extracts additional features from the stars in the given data file using
    their light curves and existing features.

    Parameters
    ---------
    data_file : str
        The file path of the input data file.
    curves_dir : str
        The directory where the light curves are stored.
    output_file : str
        The file path of where to write the output data file.
    """
    data = pd.read_csv(data_file)

    extract_func = partial(extract_with_curve, curves_dir)
    new_data = data.apply(extract_func, axis=1)

    new_data.to_csv(output_file, index=False)

def extract_with_curve(curves_dir, data):
    """
    Extracts the features from the given star's data with its light curve.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    star_id = data["Numerical_ID"]
    curve_path = get_curve_path(curves_dir, star_id)

    if path.exists(curve_path):
        curve = pd.read_csv(curve_path)

        return extract_features(data, curve, curves_dir)
    else:
        new_data = data.copy()

        new_data["lt"] = np.nan
        new_data["mr"] = np.nan
        new_data["ms"] = np.nan
        new_data["b1std"] = np.nan
        new_data["rcb"] = np.nan
        new_data["std"] = np.nan
        new_data["mad"] = np.nan
        new_data["mbrp"] = np.nan
        new_data["pa"] = np.nan
        new_data["lc_rms"] = np.nan
        new_data["lc_flux_asymmetry"] = np.nan
        new_data["sm_phase_rms"] = np.nan
        new_data["periodicity"] = np.nan
        new_data["crosses"] = np.nan

        return new_data

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

def extract_features(data, light_curve, curves_dir):
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
    light_curve : pandas.core.frame.DataFrame
        The light curve of the given star.
    curves_dir : str
        The directory that the curve files are stored in.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    new_data = data.copy()
    star_id = data["Numerical_ID"]
    period = data["Period_(days)"]

    times = light_curve.as_matrix(["MJD"])
    magnitudes = light_curve.as_matrix(["Mag"])

    phase_times = phase_fold(times, period)

    sm_phase_times, sm_phase_magnitudes = smooth_curve(phase_times, magnitudes)

    lt = linear_trend(times, magnitudes)
    mr = magnitude_ratio(magnitudes)
    ms = maximum_slope(times, magnitudes)
    b1std = beyond_1std(magnitudes)
    rcb = r_cor_bor(magnitudes)
    std = np.std(magnitudes)
    mad = median_absolute_deviation(magnitudes)
    mbrp = median_buffer_range_percentage(magnitudes)
    pa = percent_amplitude(magnitudes)

    lc_rms = root_mean_square(magnitudes)
    lc_flux_asymmetry = light_curve_flux_asymmetry(magnitudes, lc_rms)
    sm_phase_rms = root_mean_square(sm_phase_magnitudes)
    periodicity = periodicity_metric(lc_rms, sm_phase_rms)

    crosses = mean_crosses(sm_phase_magnitudes)

    new_data["lt"] = lt
    new_data["mr"] = mr
    new_data["ms"] = ms
    new_data["b1std"] = b1std
    new_data["rcb"] = rcb
    new_data["std"] = std
    new_data["mad"] = mad
    new_data["mbrp"] = mbrp
    new_data["pa"] = pa
    new_data["lc_rms"] = lc_rms
    new_data["lc_flux_asymmetry"] = lc_flux_asymmetry
    new_data["sm_phase_rms"] = sm_phase_rms
    new_data["periodicity"] = periodicity
    new_data["crosses"] = crosses

    save_curve(curves_dir, star_id, "phase", phase_times[:,0], magnitudes[:,0], ["phase", "Mag"])
    save_curve(curves_dir, star_id, "sm_phase", sm_phase_times, sm_phase_magnitudes, ["phase", "Mag"])

    return new_data

def save_curve(curves_dir, star_id, curve_name, times, magnitudes, columns):
    curve = pd.DataFrame(list(zip(times, magnitudes)), columns=columns)

    curve_path = get_curve_path(curves_dir, "%d_%s" % (star_id, curve_name))

    curve.to_csv(curve_path, index=False)

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

    for i in range(len(times) - 1):
        mag_diff = magnitudes[i + 1][0] - magnitudes[i][0]
        time_diff = times[i + 1][0] - times[i][0]

        if time_diff != 0:
            slope = mag_diff / time_diff

            if max_slope is None or slope > max_slope:
                max_slope = slope

    return max_slope

def beyond_1std(magnitudes):
    """
    Returns the percent of points in the light curve with a magnitude greater
    than one standard deviation. Values range from 0.0 to 1.0.

    bstd1 = P(|mag - mean_mag| > std)

    (D'Isanto et al., 2015) (2.1.2)

    Parameters
    ----------
    magnitudes : numpy.ndarray
        The light curve magnitudes.

    Returns
    -------
    b1std : numpy.float64
        The percent of magnitudes above 1 standard deviation.
    """
    std = np.std(magnitudes)
    mean = np.mean(magnitudes)

    gt_1_std = magnitudes[abs(magnitudes - mean) > std].size

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
    absolute_deviations = [abs(d) for d in deviations]

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
    absolute_deviations = np.array([abs(d) for d in deviations])

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

def phase_fold(times, period):
    """
    Folds the given light curve over its period to express the curve in terms
    of phase rather than time.

    Parameters
    ----------
    times : numpy.ndarray
        The light curve times.
    period : numpy.float64
        The light curve period.

    Returns
    -------
    phase_times : numpy.ndarray
        The light curve times in terms of phase.
    """
    phase_times = (times % period) / period

    return phase_times

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
    d_med = np.median(magnitudes)

    data_len = len(magnitudes)
    index_10p = int(round(data_len * 0.1))
    top_10 = data_len - index_10p - 1
    bottom_10 = index_10p
    d_10p = magnitudes[top_10:-1] + magnitudes[0:bottom_10]

    mean_d_10p = np.mean(d_10p)

    return (mean_d_10p - d_med) / light_curve_rms

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

    crosses = 0
    for mag in magnitudes:
        if above:
            if mag < dev_below:
                above = False
                crosses += 1
        else:
            if mag > dev_above:
                above = True
                crosses += 1

    return crosses

if __name__ == "__main__":
    main()
