import numpy as np
import pandas as pd
import unittest

from licubic import feature_extraction

TEST_CURVES = [
    (
        np.array([[0.0], [10.9], [12.3], [15.7], [19.2]]),
        np.array([[14.0], [14.5], [14.65], [14.4], [13.9]]),
        np.array([[0.01], [0.009], [0.02], [0.01], [0.01]])
    ),
    (
        np.array([[1.3], [2.7], [3.1], [4.6], [4.9], [5.1], [6.9], [10.2]]),
        np.array([[15.0], [15.1], [14.9], [15.3], [15.2], [15.4], [14.9], [15.0]]),
        np.array([[0.02], [0.01], [0.0093], [0.0089], [0.01], [0.015], [0.019], [0.02]])
    ),
    (
        np.array([[0.1], [9.8]]),
        np.array([[14.1], [15.9]]),
        np.array([[0.01], [0.02]])
    ),
    (
        np.array([[0.1], [1.2], [1.9], [2.1], [2.4], [2.7], [2.9], [3.33], [3.42], [3.7], [4.2], [4.5], [4.9]]),
        np.array([[14.0], [14.1], [14.3], [15.2], [16.3], [15.1], [14.2], [14.0], [13.8], [13.5], [13.9], [14.1], [14.0]]),
        np.array([[0.01], [0.01], [0.02], [0.0198], [0.012], [0.013], [0.01], [0.02], [0.0098], [0.01], [0.03], [0.01], [0.02]])
    )
]

TIMES = 0
MAGNITUDES = 1
ERRORS = 2

def try_expected(self, curves, expected_results, test):
    """
    Runs the given test function against the given curves and compares the
    results to the given expected results.
    """
    for i in range(len(curves)):
        curve = curves[i]

        expected = expected_results[i]
        actual = test(curve)

        if isinstance(expected, float) and np.isnan(expected):
            self.assertTrue(np.isnan(actual))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_allclose(actual, expected, rtol=1e-5)
        else:
            self.assertEqual(expected, actual)

class TestFeatureExtractFeatures(unittest.TestCase):
    """
    The tests defined here are regression tests used primarily to ensure that
    changes to the functions do not lead to changes in the program results.

    Each of the expected results here were caclulated using the existing code,
    and thus they are only as correct as the implementation they were calulated
    using.

    Small changes in the calculated values may be acceptable depending on the
    surrounding circumstances and nature of the change.
    """

    def test_linear_trend(self):
        expected_results = [
            0.004550040931330576,
            -0.0039444027047332722,
            0.18556701030927839,
            -0.14333805216641998
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.linear_trend(times, magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_magnitude_ratio(self):
        expected_results = [
            0.4,
            0.5,
            0.5,
            0.38461538461538464
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.magnitude_ratio(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_maximum_slope(self):
        expected_results = [
            0.10714285714285737,
            1.0000000000000089,
            0.18556701030927841,
            4.4999999999999893
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.maximum_slope(times, magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_beyond_1std(self):
        expected_results = [
            0.4,
            0.5,
            0.0,
            0.3076923076923077
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.beyond_1std(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_r_cor_bor(self):
        expected_results = [
            0.0,
            0.0,
            0.0,
            0.07692307692307693
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.r_cor_bor(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_median_absolute_deviation(self):
        expected_results = [
            0.25,
            0.14999999999999947,
            0.90000000000000036,
            0.19999999999999929
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.median_absolute_deviation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_median_buffer_range_percentage(self):
        expected_results = [
            1.0,
            1.0,
            1.0,
            0.9230769230769231
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.median_buffer_range_percentage(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_percent_amplitude(self):
        expected_results = [
            0.25,
            0.34999999999999964,
            0.90000000000000036,
            2.2000000000000011
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.percent_amplitude(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_total_variation(self):
        expected_results = [
            0.28000000000000008,
            0.20000000000000018,
            0.90000000000000036,
            0.44615384615384623
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.total_variation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_quadratic_variation(self):
        expected_results = [
            0.11700000000000003,
            0.065000000000000072,
            1.6200000000000012,
            0.36153846153846192
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.quadratic_variation(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_periodicity_metric(self):
        expected_results = [
            1.0002414654912055,
            1.0110139136470442,
            0.9976110027920946,
            0.9787365158352862
        ]

        def test(curve):
            times = curve[TIMES]
            magnitudes = curve[MAGNITUDES]
            errors = curve[ERRORS]

            period = 0.5
            phase_times, phase_magnitudes, phase_errors = \
                feature_extraction.phase_fold(times, magnitudes, errors, period)

            sm_phase_times, sm_phase_magnitudes = feature_extraction.smooth_curve(phase_times, magnitudes)

            lc_rms = feature_extraction.root_mean_square(magnitudes)
            sm_phase_rms = feature_extraction.root_mean_square(sm_phase_magnitudes)

            return feature_extraction.periodicity_metric(lc_rms, sm_phase_rms)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_light_curve_flux_asymmetry(self):
        expected_results = [
            np.nan,
            np.nan,
            np.nan,
            -0.0069616264413517003
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            lc_rms = feature_extraction.root_mean_square(magnitudes)

            return feature_extraction.light_curve_flux_asymmetry(magnitudes, lc_rms)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_root_mean_square(self):
        expected_results = [
            14.29295280898947,
            15.100993344810137,
            15.026975743641833,
            14.364459346167273
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.root_mean_square(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_chi_2_test(self):
        expected_results = [
            5823845.9725812068,
            9882084.1746215299,
            659891.24999999988,
            14702747.809515834
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            errors = curve[ERRORS]

            return feature_extraction.chi_2_test(magnitudes, errors)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_interquartile_range(self):
        expected_results = [
            -0.59999999999999964,
            -1.2999999999999989
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            return feature_extraction.interquartile_range(magnitudes)

        try_expected(self, [TEST_CURVES[0]] + [TEST_CURVES[3]], expected_results, test)

    """def test_robust_median_statistic(self):
        expected_results = [
            np.nan,
            np.nan,
            np.nan,
            np.nan
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            errors = curve[ERRORS]
            return feature_extraction.robust_median_statistic(magnitudes, errors)

        try_expected(self, TEST_CURVES[0], expected_results, test)
        """

    def test_peak_to_peak_variability(self):
        expected_results = [
            0.025227750525578158,
            0.01570262392595306,
            0.059019673224408174,
            0.093227733404926516
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            errors = curve[ERRORS]

            return feature_extraction.peak_to_peak_variability(magnitudes, errors)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_fourier_decomposition(self):
        expected_results = [
            np.array([4.44354063, 3.48308428, -2.47321523, 4.71882566, -0.18528359, 3.46454684, 1.74425983]),
            np.array([2.548879, 0.750577, 5.5012, 1.516458, 1.09882, 1.035017, 3.423319])
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            times = curve[TIMES]
            order = 3

            return feature_extraction.fourier_decomposition(magnitudes, times, order)

        try_expected(self, TEST_CURVES[1:2] + TEST_CURVES[3:], expected_results, test)

    def test_fourier_decomposition_few_examples(self):
        curve = TEST_CURVES[0]
        magnitudes = curve[MAGNITUDES]
        times = curve[TIMES]
        order = 3

        try:
            feature_extraction.fourier_decomposition(magnitudes, times, order)
            self.assertTrue(False)
        except Exception as e:
            expected = "Too few examples for the specified order. Number of examples must be at least order * 2 + 1. Required: 7, Actual: 5"
            actual = str(e)
            self.assertEqual(expected, actual)

    def test_fourier_R_1(self):
        expected_results = [
            np.array([1.354784, 0.994678]),
            np.array([2.02039, 1.378962])
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            times = curve[TIMES]
            order = 3

            coef = feature_extraction.fourier_decomposition(magnitudes, times, order)

            r_21 = feature_extraction.fourier_R_1(coef, 2)
            r_31 = feature_extraction.fourier_R_1(coef, 3)

            return np.array([r_21, r_31])

        try_expected(self, TEST_CURVES[1:2] + TEST_CURVES[3:], expected_results, test)

    def test_fourier_phi_1(self):
        expected_results = [
            np.array([4.761147, 2.88072]),
            np.array([2.662789, 5.769273])
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            times = curve[TIMES]
            order = 3

            coef = feature_extraction.fourier_decomposition(magnitudes, times, order)

            phi_21 = feature_extraction.fourier_phi_1(coef, 2)
            phi_31 = feature_extraction.fourier_phi_1(coef, 3)

            return np.array([phi_21, phi_31])

        try_expected(self, TEST_CURVES[1:2] + TEST_CURVES[3:], expected_results, test)

    def test_residual_bright_faint_ratio(self):
        expected_results = [
            1.9068891280947429,
            0.42857142857143882,
            1.0,
            0.10073344824793815
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            return feature_extraction.residual_bright_faint_ratio(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_cumulative_sum_range(self):
        expected_results = [
            57.450000000000003,
            105.80000000000001,
            15.9,
            172.5
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            return feature_extraction.cumulative_sum_range(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_von_neumann_eta(self):
        expected_results = [
            1.7328199052132705,
            2.4761904761904781,
            4.0,
            0.74523380619980528
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            return feature_extraction.von_neumann_eta(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_mean_crosses(self):
        expected_results = [
            2,
            2,
            1,
            2
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            return feature_extraction.mean_crosses(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_above_1std(self):
        expected_results = [
            0.2,
            0.25,
            0.0,
            0.23076923076923078
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]

            return feature_extraction.above_1std(magnitudes)

        try_expected(self, TEST_CURVES, expected_results, test)

    def test_above_below_1std_slopes(self):
        expected_results = [
            (0.25, 0.25),
            (0.14285714285714285, 0.14285714285714285),
            (0.0, 0.0),
            (0.16666666666666666, 0.16666666666666666)
        ]

        def test(curve):
            magnitudes = curve[MAGNITUDES]
            times = curve[TIMES]

            slopes = feature_extraction.curve_slopes(times, magnitudes)

            return feature_extraction.above_below_1std_slopes(slopes)

        try_expected(self, TEST_CURVES, expected_results, test)

#    def test_extract_features(self):
#        star_id_col = "id"
#        period_col = "period"
#        data = pd.DataFrame(["jne2e"])
#        data.columns = [star_id_col]
#        data[period_col] = 0.2342
#        data = feature_extraction.add_feature_columns(data)
#        data = data.iloc[0]

#        curves_dir = ""
#        save_curve_files = False

#        times = np.arange(0, 10.0, 0.5)
#        magnitudes = np.sin(times) + 14.0
#        errors = times / 100.0
#        light_curve = np.transpose(np.vstack([times, magnitudes, errors]))

#        feature_extraction.extract_features(data, star_id_col, period_col, light_curve, curves_dir, save_curve_files)

class TestFeatureExtractUtilities(unittest.TestCase):

    def test_clean_light_curve_ordered(self):
        """
        An already ordered light curve with no duplicate observations should be
        returned the same.
        """
        times = np.array([[0.0], [1.3], [2.9], [3.0]])
        magnitudes = np.array([[14.2], [14.0], [13.9], [14.2]])
        errors = np.array([[0.005], [0.01], [0.02], [0.001]])

        act_time, act_mags, act_err = feature_extraction.clean_light_curve(times, magnitudes, errors)

        np.testing.assert_allclose(act_time, times, rtol=1e-5)
        np.testing.assert_allclose(act_mags, magnitudes, rtol=1e-5)
        np.testing.assert_allclose(act_err, errors, rtol=1e-5)

    def test_clean_light_curve_unsorted(self):
        """
        An unsorted light curve should be returned sorted.
        """
        times = np.array([[3.9], [1.3], [4.9], [3.0]])
        magnitudes = np.array([[14.2], [14.0], [13.9], [14.2]])
        errors = np.array([[0.005], [0.01], [0.02], [0.001]])

        act_time, act_mags, act_err = feature_extraction.clean_light_curve(times, magnitudes, errors)

        exp_time = np.array([[1.3], [3.0], [3.9], [4.9]])
        exp_mags = np.array([[14.0], [14.2], [14.2], [13.9]])
        exp_err = np.array([[0.01], [0.001], [0.005], [0.02]])

        np.testing.assert_allclose(act_time, exp_time, rtol=1e-5)
        np.testing.assert_allclose(act_mags, exp_mags, rtol=1e-5)
        np.testing.assert_allclose(act_err, exp_err, rtol=1e-5)

    def test_clean_light_curve_duplicates(self):
        """
        An unsorted light curve with duplicate should be returned sorted
        without the duplicate observation.
        """
        times = np.array([[1.3], [1.3], [4.9], [3.0]])
        magnitudes = np.array([[14.2], [14.2], [13.9], [14.2]])
        errors = np.array([[0.01], [0.01], [0.02], [0.001]])

        act_time, act_mags, act_err = feature_extraction.clean_light_curve(times, magnitudes, errors)

        exp_time = np.array([[1.3], [3.0], [4.9]])
        exp_mags = np.array([[14.2], [14.2], [13.9]])
        exp_err = np.array([[0.01], [0.001], [0.02]])

        np.testing.assert_allclose(act_time, exp_time, rtol=1e-5)
        np.testing.assert_allclose(act_mags, exp_mags, rtol=1e-5)
        np.testing.assert_allclose(act_err, exp_err, rtol=1e-5)

np.random.seed(42)
TEST_GEN_CURVES = [
    np.array([
        np.arange(0, 10, 0.5),
        np.sin(np.arange(0, 10, 0.5)),
        np.random.normal(0.03, 0.01, 20)
    ]).transpose(),
    np.array([
        np.arange(0, 10, 0.5),
        np.cos(np.arange(0, 10, 0.5)),
        np.random.normal(0.03, 0.01, 20)
    ]).transpose(),
    np.array([
        np.arange(0, 20, 0.5),
        np.sin(np.arange(0, 20, 0.5) * np.pi),
        np.random.normal(0.03, 0.01, 40)
    ]).transpose()
]

class TestFeatureExtractor(unittest.TestCase):

    def test_transform(self):
        extractor = feature_extraction.FeatureExtractor()

        actual = extractor.transform(TEST_GEN_CURVES)

        expected = np.array([
            [0.987513, -0.002155, 0.5, 0.989071, 0.4, 0.0, 0.56044324, 0.0, 0.68387575, 9.0, 2.01884346, -1.33656453, 1.53827137e-01, 2.51539805e-01, 0.41648757, 0.64386049, 0.80706132, -1.59462679e-01],
            [0.998586, -0.057269, 0.5, 0.988916, 0.5, 0.0, 0.75490669, 0.0, 1.03738142, 12.0, 2.08331462, -1.8059096, 2.70091119e-01, 5.10890308e-01, 0.66092537, 0.85388423, 0.94498811, 4.08641285e-02],
            [1.0, -0.007505, 0.5, 2.0, 0.5, 0.0, 0.5, 0.0, 1.0, 16.0, 2.11377926, -1.08179232, 8.40370077e-16, 1.78578641e-15, 0.25, 1.0, 1.0, 1.18686293e-15]
        ])

        print(actual[:,-1])
        print(expected[:,-1])

        np.testing.assert_allclose(actual, expected, rtol=1e-4)
