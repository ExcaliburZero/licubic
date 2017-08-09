import compairisons

import itertools
import numpy as np
import pandas as pd
import pickle
import pystache
import sys
import datetime

RED_CELL_UPPER = 0.50
YELLOW_CELL_UPPER = 0.75

def main():
    data_file = sys.argv[1]
    category_col = "category"
    matrix_file = "site/index.html"

    category_minimum = 70
    balanced = False

    # List incomplete and should be changed as needed
    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr", "fpr20", "fpr35", "fpr50", "fpr65", "fpr80"
        ,  "roms", "ptpv", "skewness", "kurtosis", "ampl", "stetson_I", "stetson_J", "stetson_K", "pst", "pdfp", "sk"
        ,  "cum_sum"
        ,  "neumann_eta", "residual_br_fa_ratio", "shapiro_wilk", "slopes_10per"
        ,  "slopes_90per", "abv_1std", "abv_1std_slopes", "bel_1std", "bel_1std_slopes"
        ]

    data = pd.read_csv(data_file)
    data = data[[category_col] + features_cols]

    data = remove_partial_entries(data)

    data, good_categories = remove_small_categories(data, category_col, features_cols, category_minimum)

    matrix, classifiers = compairisons.feature_matrix(category_col, features_cols, data, balanced=balanced)

    write_html_matrix(matrix, good_categories, matrix_file)

    date_time = str(datetime.datetime.now())

    features_file = "data/best_features_" + date_time + ".txt"
    save_best_features(matrix, features_cols, features_file)

    classifiers_file = "data/classifiers_" + date_time + ".pickle"
    save_classifiers(classifiers_file, classifiers)

def remove_partial_entries(data):
    size_before = len(data)
    data = data.dropna()
    size_after = len(data)

    if size_before > size_after:
        print("%d points removed due to missing feature values." % (size_before - size_after))

    return data

def remove_small_categories(data, category_col, features_cols, category_minimum):
    categories = data[category_col].unique()

    category_info = data.groupby(category_col).count()[features_cols[0]]
    good_categories = category_info.index[np.array(category_info) > category_minimum]

    good_categories = [x for x in categories if x in good_categories]

    num_removed_categories = len(categories) - len(good_categories)
    if num_removed_categories > 0:
        print("Removed %d categories with less than %d instances." % (num_removed_categories, category_minimum))

    data = data[data[category_col].isin(good_categories)]

    return data, good_categories

def save_classifiers(classifiers_file, classifiers):
    with open(classifiers_file, "wb") as f:
        pickle.dump(classifiers, f)

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
            comb = (a, b)
            if comb in matrix:
                features = np.array(matrix[comb][0])[:,0]
                cm_1 = matrix[comb][1]
                cm_2 = matrix[comb][2]
                score = matrix[comb][5]

                cell_color = "cell-"
                if score < RED_CELL_UPPER:
                    cell_color += "red"
                elif score < YELLOW_CELL_UPPER:
                    cell_color += "yellow"
                else:
                    cell_color += "green"

                table += "<td class='" + cell_color + "'>"

                table += "<div>"
                table += create_confusion_matrix(cm_1)
                table += create_confusion_matrix(cm_2)
                table += "</div>"

                a_examples = matrix[comb][3]
                b_examples = matrix[comb][4]

                table += "<div>"
                table += "%d ~ %d" % (a_examples, b_examples)
                table += "<br />%f" % score
                table += "</div>"

                table += "<br />".join(features)
            else:
                table += "<td>"

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

def save_best_features(matrix, features_cols, filepath):
    contents = list_best_features(matrix, features_cols)

    write_file(filepath, contents)

if __name__ == "__main__":
    main()
