import collections
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotnine as plt9
import pystache
import scikitplot as sciplt
import seaborn as sns
import sklearn.metrics
import sys

from . import compairisons
from . import feature_extraction

RED_CELL_UPPER = 0.50
YELLOW_CELL_UPPER = 0.75

def main():
    classifiers_file = sys.argv[1]
    data_file = sys.argv[2]

    classifier = create_classifier(classifiers_file)

    data = pd.read_csv(data_file)

    features_cols = ["lt", "mr", "ms", "b1std", "rcb", "std", "mad", "mbrp"
        ,  "pa", "lc_flux_asymmetry", "chi_2", "iqr", "fpr20", "fpr35", "fpr50", "fpr65", "fpr80"
        ,  "roms", "ptpv", "skewness", "kurtosis", "ampl", "stetson_I", "stetson_J", "stetson_K", "pst", "pdfp", "sk"
        ,  "cum_sum"
        ,  "neumann_eta", "residual_br_fa_ratio", "shapiro_wilk", "slopes_10per"
        ,  "slopes_90per", "abv_1std", "abv_1std_slopes", "bel_1std", "bel_1std_slopes"
        ]

    category_col = "category"
    data = data[features_cols + [category_col]]

    data = remove_partial_entries(data)

    category_minimum = 0
    data = remove_small_categories(data, category_col, features_cols, category_minimum)

    predictions = classifier.predict(data)

    y = np.array(data[category_col])
    y_pred = predictions

    date_time = str(datetime.datetime.now())
    filepath = "cm_%s.svg" % date_time
    save_confusion_matrix(y, y_pred, filepath, data, category_col, classifier)

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

    return data

def save_confusion_matrix(y, y_pred, filepath, data, category_col, classifier):
    true_cats = np.array(data[category_col].unique())
    pred_cats = np.array(classifier.categories + [classifier.unknown])

    labels = np.sort(np.unique(np.concatenate([true_cats, pred_cats])))
    true_labels = np.where(np.isin(labels, true_cats))
    pred_labels = np.where(np.isin(labels, pred_cats))

    #sns.set_context("talk", font_scale=2.2)
    #figsize = (54, 38)
    #figsize = (20, 16)
    figsize = (26, 18)
    fontsize = 20
    sns.set_context("poster")
    #figsize = (20, 16)
    #fontsize = 20

    title = "Confusion Matrix ~ CRTS North -> CRTS South"

    #normalize = "percent"
    normalize = True
    ax = sciplt.plotters.plot_confusion_matrix(y, y_pred, hide_zeros=True, normalize=normalize, x_tick_rotation=90, title_fontsize="large", text_fontsize="large", true_label_indexes=true_labels, pred_label_indexes=pred_labels, labels=labels, figsize=figsize, title=title)
    for text in ax.texts:
        #text.set_weight('bold')
        #text.set_fontsize(80)
        text.set_fontsize(fontsize * 1.2)
    for text in ax.xaxis.get_ticklabels():
        text.set_fontsize(fontsize)
    #    text.set_weight('bold')
    for text in ax.yaxis.get_ticklabels():
        text.set_fontsize(fontsize)
    #    text.set_weight('bold')
    #ax.xaxis.label.set_weight('bold')
    #ax.yaxis.label.set_weight('bold')
    #ax.title.set_weight('bold')
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.title.set_fontsize(fontsize)
    #plt.show()
    plt.savefig(filepath)

def create_classifier(classifiers_file):
    binary_classifiers = load_binary_classifiers(classifiers_file)

    a_vs_not_a = get_a_versus_not_a(binary_classifiers)

    classifier = BlackBoxClassifier(a_vs_not_a)

    return classifier

def load_binary_classifiers(classifiers_file):
    with open(classifiers_file, "rb") as f:
        return pickle.load(f)

def get_a_versus_not_a(binary_classifiers):
    binaries = binary_classifiers.keys()
    wanted_binaries = [(a, b) for (a, b) in binaries if a == b]

    a_vs_not_a = collections.OrderedDict({})
    for binary in wanted_binaries:
        classifier = binary_classifiers[binary]
        a_vs_not_a[binary] = classifier

    key_order = sorted(a_vs_not_a.keys())
    a_vs_not_a = collections.OrderedDict((k, a_vs_not_a[k]) for k in key_order)

    return a_vs_not_a

class LICUBIC(object):

    def __init__(self, features=None):
        self.unknown_threshold = 0.50
        self.unknown = "unknown"

        self.trained = False
        self.categories = None

        self.feature_extractor_train = feature_extraction.FeatureExtractor(features=features)

    def train(self, X, y):
        y = np.array(y)

        self.trained = True

        if self.unknown in y:
            raise Exception("There cannot be a category named %s as this name is reserved." % self.unknown)

        X_feat = self.feature_extractor_train.transform(X)
        external_features = self.feature_extractor_train.get_external_features()

        self.matrix, self.binary_classifiers = compairisons.feature_matrix(X_feat, y, external_features)

        self.a_vs_not_a = get_a_versus_not_a(self.binary_classifiers)
        self.categories = [a for (a, _) in self.a_vs_not_a.keys()] 

        self.selected_features = collections.OrderedDict()
        for (features, _) in self.a_vs_not_a.values():
            for f in features:
                self.selected_features[f] = True

        self.feature_extractor_predict = self.feature_extractor_train.subset(self.selected_features)

    def predict(self, X):
        results = self.get_class_probabilities(X)

        predictions = []
        for r in results:
            best_match = np.argmax(r)
            best_score = r[best_match]

            if best_score > self.unknown_threshold:
                pred = self.categories[best_match]
            else:
                pred = self.unknown

            predictions.append(pred)

        return np.array(predictions)

    def get_class_probabilities(self, X):
        X_features = self.feature_extractor_predict.transform(X)

        ordered_sel_features = list(self.feature_extractor_predict.get_external_features())

        results = []
        for (features, classifier) in self.a_vs_not_a.values():
            feature_indexes = [ordered_sel_features.index(f) for f in features]

            X_feat = X_features[:,feature_indexes]

            res = classifier.predict_proba(X_feat)
            results.append(res)

        results = np.array(results)

        results = results[:,:,1]
        results = np.swapaxes(results, 0, 1)

        return results

    def write_confusion_matrix(self, y, y_pred, save_file_path):
        true_labels = np.unique(y)
        pred_labels = np.array(self.categories + [self.unknown])

        labels = np.sort(np.unique(np.concatenate([true_labels, pred_labels])))
        #true_labels = np.where(np.isin(labels, true_cats))
        #pred_labels = np.where(np.isin(labels, pred_cats))

        figsize = (26, 18)
        fontsize = 20
        sns.set_context("poster")

        title = "Confusion Matrix ~ CRTS North -> CRTS South"

        normalize = True
        ax = sciplt.plotters.plot_confusion_matrix(y, y_pred, hide_zeros=True, normalize=normalize, x_tick_rotation=90, title_fontsize="large", text_fontsize="large", true_labels=true_labels, pred_labels=pred_labels, labels=labels, figsize=figsize, title=title)

        for text in ax.texts:
            text.set_fontsize(fontsize * 1.2)
        for text in ax.xaxis.get_ticklabels():
            text.set_fontsize(fontsize)
        for text in ax.yaxis.get_ticklabels():
            text.set_fontsize(fontsize)

        ax.xaxis.label.set_fontsize(fontsize)
        ax.yaxis.label.set_fontsize(fontsize)
        ax.title.set_fontsize(fontsize)
        plt.savefig(save_file_path)

    def write_feature_matrix(self, save_file_path):
        write_html_matrix(self.matrix, self.categories, save_file_path)


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

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
