import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystache
import scikitplot as sciplt
import seaborn as sns
import sklearn.metrics

from . import compairisons
from . import feature_extraction

RED_CELL_UPPER = 0.50
YELLOW_CELL_UPPER = 0.75

UNKNOWN_THRESHOLD = 0.50
UNKNOWN_CATEGORY = "unknown"

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
        self.unknown_threshold = UNKNOWN_THRESHOLD
        self.unknown = UNKNOWN_CATEGORY

        self.trained = False
        self.categories = None

        self.feature_extractor_train = feature_extraction.FeatureExtractor(features=features)

    def fit(self, X, y):
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
