## Installation

```
$ virtualenv -p /usr/bin/python3.5 tran_class_features --no-site-packages
$ source activate tran_class_features
$ pip install -r requirements.txt
$ mkdir data
```

## Usage

### Training Classifiers and Feature Matrix
To train the set of binary classifiers and create the html feature matrix, run the following command substituting in the proper csv data file.

```
$ python transient_classification_features/__main__.py DATA_FILE
```

For example, if the data file is `crts_plus_trans.csv` you would run the following.

```
$ python transient_classification_features/__main__.py crts_plus_trans.csv
```

The feature matrix will be written to `site/index.html` and the classifier set will be saved in the `data` folder.

### Running Black Box Classifier
To run a Black Box Classifier using a set of trained binary classifiers, run the following command substituting in the proper csv data file.

```
$ python transient_classification_features/classifier.py CLASSIFIER_SET DATA_FILE
```

For example, if the classifier set file is `data/classifiers_2017-08-01 18:41:58.711333.pickle` and the data file is `~/Datasets/sss/SSS_Per_Tab_extracted.csv` you fould run the following.

```
$ python transient_classification_features/classifier.py "data/classifiers_2017-08-01 18:41:58.711333.pickle" ~/Datasets/sss/SSS_Per_Tab_extracted.csv
```

The confusion matrix and the list of best overall features will be saved in the `data` directory.

## Bugs / Tweaks
* There is currently a [bug](https://github.com/scikit-learn/scikit-learn/issues/7346) in scikit-learn (v 0.18.1) that makes `RandomForestClassifier`s not threadsafe. This causes issues when multiple classifiers are trained in paralel,specifically it raises the following exception `IndexError: pop from empty list`. The current workaround needed for this application is to edit the `sklearn/base.py` file and replace line 248 with `pass`.
* For the classifier you will need a patched version of `scikit-plot` that includes some special features. You can install the patched version by running the following commands.
```
$ cd ..
$ git clone git@github.com:ExcaliburZero/scikit-plot.git
$ cd scikit-plot
$ git checkout cm-nan
$ pip install ./ -U
```
