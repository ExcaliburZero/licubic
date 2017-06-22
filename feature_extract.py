from functools import partial
from os import path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn import linear_model

import argparse
import math
import numpy as np
import pandas as pd
import sys

def main():
    """
    $ python3 feature_extract.py CatalinaVars.csv curves/ test.csv
    """
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    result = validate_arguments(args)

    if isinstance(result, str):
        print(result, file=sys.stderr)
        sys.exit(1)

    main_functionality(
            args.data_file,
            args.curves_dir,
            args.output_file,
            nrows=args.nrows,
            save_curve_files=args.save_curve_files
        )

def create_arg_parser():
    """
    Creates and returns the command line arguments parser for the script.

    Returns
    -------
    arg_parser : argparse.ArgumentParser
        The command line argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Extract features from CRTS light curve data.")

    # Required arguments
    parser.add_argument("data_file", type=str,
            help="the input data file")
    parser.add_argument("curves_dir", type=str,
            help="the directory where the light curves are stored")
    parser.add_argument("output_file", type=str,
            help="the output data file")

    # Optional flags
    parser.add_argument("--nrows", dest="nrows", type=int, default=None,
            help="the number of rows of data to process (default: all)")
    parser.add_argument("--save-curves", dest="save_curve_files", action="store_const",
            const=True, default=False,
            help="save the intermediate light curves")

    return parser

def validate_arguments(args):
    """
    Checks to see if the given command line arguments are valid. Returns None
    if they are all valid, or an error string if one or more are invalid.

    Parameters
    ---------
    args : argparse.Namespace
        The parsed command line arguments.

    Returns
    -------
    error : Union[None, str]
        The error message if at least one argument was invalid, otherwise is
        None.
    """
    data_file = args.data_file
    curves_dir = args.curves_dir
    nrows = args.nrows

    if not path.exists(data_file):
        return "The given data file does not exist: %s" % data_file
    if not path.isfile(data_file):
        return "The given data file is not a file: %s" % data_file

    if not path.exists(curves_dir):
        return "The given curve file directory does not exist: %s" % curves_dir
    if not path.isdir(curves_dir):
        return "The given curve file directory is not a directory: %s" % curves_dir

    if nrows is not None:
        if nrows < 0:
            return "The given nrows is not a non-negative integer: %s" % nrows

    return None

def main_functionality(data_file, curves_dir, output_file, nrows=None,
        save_curve_files=False):
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
    nrows : Union[int, None]
        The number of rows to process from the data file. If None, then
        processes all of the rows.
    save_curve_files : bool
        If True, then the intermediate light curves are saved to the
        curves_dir.
    """
    data = pd.read_csv(data_file, nrows=nrows)

    columns = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "totvar", "quadvar", "fslope", "lc_rms"
        ,  "lc_flux_asymmetry", "sm_phase_rms", "periodicity"
        ,  "crosses", "abv_1std", "bel_1std", "abv_1std_slopes"
        ,  "bel_1std_slopes"
        ]
    data = pd.concat([data, pd.DataFrame(columns=columns)])

    extract_func = partial(extract_with_curve, curves_dir, save_curve_files)
    new_data = data.apply(extract_func, axis=1)

    new_data.to_csv(output_file, index=False)

def extract_with_curve(curves_dir, save_curve_files, data):
    """
    Extracts the features from the given star's data with its light curve.

    Parameters
    ----------
    curves_dir : str
        The directory that the curve files are stored in.
    save_curve_files : bool
        If True, then the intermediate light curves are saved to the
        curves_dir.
    data : pandas.core.frame.DataFrame
        The exisiting data on the given star.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        The existing and extracted information on the given star.
    """
    star_id = int(data["Numerical_ID"])
    curve_path = get_curve_path(curves_dir, star_id)

    if path.exists(curve_path):
        curve = get_curve(curve_path)

        return extract_features(data, curve, curves_dir, save_curve_files)
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

    This function exists on its own in order to make the script easier to
    profile, so that the time it takes to load the light curves is specifically
    measured.

    Parameters
    ----------
    curve_path : str
        The file path of the curve file for the given star.

    """
    return pd.read_csv(curve_path)

def extract_features(data, light_curve, curves_dir, save_curve_files):
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
        ,  "lc_flux_asymmetry", "sm_phase_rms", "periodicity"
        ,  "crosses", "abv_1std", "bel_1std", "abv_1std_slopes"
        ,  "bel_1std_slopes"
        ]

    star_id = data["Numerical_ID"]
    period = data["Period_(days)"]

    times = light_curve.as_matrix(["MJD"])
    magnitudes = light_curve.as_matrix(["Mag"])

    phase_times = phase_fold(times, period)

    sm_phase_times, sm_phase_magnitudes = smooth_curve(phase_times, magnitudes)

    sm_phase_slopes = curve_slopes(sm_phase_times, sm_phase_magnitudes)

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

    crosses = mean_crosses(sm_phase_magnitudes)
    abv_1std = above_1std(sm_phase_magnitudes)
    bel_1std = beyond_1std(sm_phase_magnitudes) - abv_1std

    abv_1std_slopes, bel_1std_slopes = above_below_1std_slopes(sm_phase_slopes)

    new_data[columns] = [lt, mr, ms, b1std, rcb, std, mad, mbrp
        ,  pa, totvar, quadvar, fslope, lc_rms
        ,  lc_flux_asymmetry, sm_phase_rms, periodicity
        ,  crosses, abv_1std, bel_1std, abv_1std_slopes, bel_1std_slopes
        ]

    if save_curve_files:
        save_curve(curves_dir, star_id, "phase", phase_times, magnitudes, ["phase", "Mag"])
        save_curve(curves_dir, star_id, "sm_phase", sm_phase_times, sm_phase_magnitudes, ["phase", "Mag"])

    return new_data

def save_curve(curves_dir, star_id, curve_name, times, magnitudes, columns):
    curve = pd.DataFrame(list(zip(times[:,0], magnitudes[:,0])), columns=columns)

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

    mag_diffs = magnitudes[1:] - magnitudes[0:-1]
    time_diffs = times[1:] - times[0:-1]

    slopes = np.divide(mag_diffs, time_diffs)

    max_slope = np.max(slopes[slopes != float("+inf")])

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

if __name__ == "__main__":
    main()
