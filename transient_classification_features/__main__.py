import compairisons

import itertools
import numpy as np
import pandas as pd
import sys

def main():
    data_file = sys.argv[1]
    category_col = "category"

    # List incomplete and should be changed as needed
    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr"
        ,  "roms", "ptpv", "skewness", "kurtosis"
        ,  "abv_1std", "bel_1std"
        ]

    data = pd.read_csv(data_file)
    data = data[[category_col] + features_cols].dropna()

    matrix = compairisons.feature_matrix(category_col, features_cols, data)

    #print_matrix(matrix)

    categories = data[category_col].unique()
    print(html_matrix(matrix, categories))

def print_matrix(matrix):
    for key in matrix:
        entry = matrix[key]
        a = key[0]
        b = key[1]
        line = ""
        if a == b:
            line += a + " vs ~" + a
        else:
            line += a + " vs " + b
        line += " = " + str(entry[1]) + "\n"

        for (feature, importance) in entry[0]:
            line += "\t" + feature + " = " + str(importance) + "\n"

        print(line)

def html_matrix(matrix, categories):
    table = "<table>"
    table += "<tr>"
    table += "<th></th>"
    for a in categories:
        table += "<th>" + a + "</th>"
    table += "</tr>"

    for a in categories:
        table += "<tr>"
        table += "<th>" + a + "</th>"
        for b in categories:
            table += "<td>"

            comb = (a, b)
            if comb in matrix:
                #table += str(matrix[comb])
                features = np.array(matrix[comb][0])[:,0]
                score = matrix[comb][1]

                table += "%.2f" % score + "<br />"
                table += "<br />".join(features)

            table += "</td>"
        table += "</tr>"
    table += "</table>"

    return table

if __name__ == "__main__":
    main()
