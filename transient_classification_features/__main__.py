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
        ]

    data = pd.read_csv(data_file)
    data = data[[category_col] + features_cols]

    size_before = len(data)
    data = data.dropna()
    size_after = len(data)

    if size_before > size_after:
        print("%d points removed due to missing feature values." % (size_before - size_after))

    matrix = compairisons.feature_matrix(category_col, features_cols, data)

    categories = data[category_col].unique()
    print(html_matrix(matrix, categories))

    print("")
    print("")
    print(list_best_features(matrix))

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
    table = "<table id='fixed' class='fancyTable'>"
    table += "<thead>"
    table += "<tr>"
    table += "<td></td>"
    for a in categories:
        table += "<th>" + a + "</th>"
    table += "</tr>"
    table += "</thead>"

    table += "<tbody>"
    for a in categories:
        table += "<tr>"
        table += "<th>" + a + "</th>"
        for b in categories:
            table += "<td>"

            comb = (a, b)
            if comb in matrix:
                #table += str(matrix[comb])
                features = np.array(matrix[comb][0])[:,0]
                score_1 = matrix[comb][1]
                score_2 = matrix[comb][2]

                table += "%.2f ~ %.2f" % (score_1, score_2) + "<br />"
                table += "<br />".join(features)

            table += "</td>"
        table += "</tr>"
    table += "</tbody>"
    table += "</table>"

    return table

def list_best_features(matrix):
    lines = ""
    for (f,v) in compairisons.rank_features(matrix):
        lines += f + " = " + str(v) + "\n"

    return lines

if __name__ == "__main__":
    main()
