## Installation

```
$ virtualenv -p /usr/bin/python3.5 tran_class_features --no-site-packages
$ source activate tran_class_features
$ pip install -r requirements.txt
```

## Usage

### Training Classifiers and Feature Matrix
To train the set of binary classifiers and create the html feature matrix, run the following command substituting in the proper csv data file.

```
$ python3 transient_classification_features/__main__.py DATA_FILE
```

For example, if the data file is `crts_plus_trans.csv` you would run the following.

```
$ python3 transient_classification_features/__main__.py crts_plus_trans.csv
```

The feature matrix will be written to `site/index.html` and the classifier set will be saved in the `data` folder.
