import compairisons

import itertools
import numpy as np
import pandas as pd
import pystache
import sys

def main():
    data_file = sys.argv[1]
    category_col = "category"
    matrix_file = "site/index.html"

    # List incomplete and should be changed as needed
    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr"
        ,  "roms", "ptpv", "skewness", "kurtosis", "ampl", "stetson_I", "stetson_K"
        ,  "cum_sum"
        ,  "neumann_eta", "residual_br_fa_ratio", "shapiro_wilk", "slopes_10per"
        ,  "slopes_90per", "abv_1std", "abv_1std_slopes", "bel_1std", "bel_1std_slopes"
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
    write_html_matrix(matrix, categories, matrix_file)

    print("")
    print("")
    print(list_best_features(matrix, features_cols))

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

def write_html_matrix(matrix, categories, output_file):
    table = html_matrix(matrix, categories)

    template_file = "site/index.mustache"
    template = read_file(template_file)

    renderer = pystache.Renderer(escape=lambda u: u)
    html = renderer.render(template, {"table": table}, escape=lambda u: u)

    write_file(output_file, html)

def write_file(f, contents):
    with open(f, "w") as myfile:
        myfile.write(contents)

def read_file(f):
    with open(f, "r") as myfile:
        contents = myfile.read()

    return contents

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
                features = np.array(matrix[comb][0])[:,0]
                score_1 = matrix[comb][1]
                score_2 = matrix[comb][2]

                table += "<div>"
                table += create_confusion_matrix(score_1)
                table += create_confusion_matrix(score_2)
                table += "</div>"

                table += "<br />".join(features)

            table += "</td>"
        table += "</tr>"
    table += "</tbody>"
    table += "</table>"

    return table

def create_confusion_matrix(cm):
    table = ""
    table += "<div class='cm'>"
    table += "<div class='cm-row'>"
    table += "<div class='cm-cell' " + bg_color(cm[0][0])  + ">%.2f</div>" % cm[0][0]
    table += "<div class='cm-cell' " + bg_color(cm[0][1])  + ">%.2f</div>" % cm[0][1]
    table += "</div>"
    table += "<div class='cm-row'>"
    table += "<div class='cm-cell' " + bg_color(cm[1][0])  + ">%.2f</div>" % cm[1][0]
    table += "<div class='cm-cell' " + bg_color(cm[1][1])  + ">%.2f</div>" % cm[1][1]
    table += "</div>"
    table += "</div>"

    return table

def bg_color(value):
    color_range = 70

    color_inc = 255 - color_range
    color = int((1.0 - value) * color_range + color_inc)

    if color < (255 / 2):
        text_color = 255
    else:
        text_color = 0

    background_color = "background-color: rgb(%d, %d, %d);" % (color, color, color)
    text_color = "color: rgb(%d, %d, %d)" % (text_color, text_color, text_color)

    return "style='" + background_color + text_color + "'"

def list_best_features(matrix, features):
    lines = ""
    for (f,v) in compairisons.rank_features(matrix, features):
        lines += f + " = " + str(v) + "\n"

    return lines

if __name__ == "__main__":
    main()
