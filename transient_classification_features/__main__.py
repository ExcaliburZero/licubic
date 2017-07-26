import compairisons

import pandas as pd
import sys

def main():
    data_file = sys.argv[1]
    category_col = "category"

    # List incomplete and should be changed as needed
    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr"
        ,  "roms", "ptpv", "fourier_amplitude", "R_21", "R_31", "f_phase"
        ,  "phi_21", "phi_31", "skewness", "kurtosis"
        ,  "abv_1std", "bel_1std"
        ]

    data = pd.read_csv(data_file)
    data = data[[category_col] + features_cols].dropna()

    compairisons.feature_matrix(category_col, features_cols, data)

if __name__ == "__main__":
    main()
