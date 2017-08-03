# Light Curve Features

## Installation

```
$ virtualenv -p /usr/bin/python3.5 lc_features --no-site-packages
$ source activate lc_features
$ pip install -r requirements.txt
```

## Usage
```
$ python light_curve_features/__main__.py INPUT_FILE CURVES_DIR OUTPUT_FILE
```

Example usage:
```
$ python light_curve_features/__main__.py ~/Datasets/eros_2/VariablesInEROS_for_StarDB.csv ~/Datasets/eros_2/curves VariablesInEROS_for_StarDB_extracted.csv --star-id-col EROSID --period-col Period --category-col VARIABLE_CLASS
```

### Options

```
usage: __main__.py [-h] [--nrows NROWS] [--save-curves]
                   [--star-id-col STAR_ID_COL] [--period-col PERIOD_COL]
                   [--category-col CATEGORY_COL] [--time-col TIME_COL]
                   [--mag-col MAG_COL] [--err-col ERR_COL]
                   data_file curves_dir output_file

Extract features from CRTS light curve data.

positional arguments:
  data_file             the input data file
  curves_dir            the directory where the light curves are stored
  output_file           the output data file

optional arguments:
  -h, --help            show this help message and exit
  --nrows NROWS         the number of rows of data to process (default: all)
  --save-curves         save the intermediate light curves
  --star-id-col STAR_ID_COL
                        the name of the column that contains the star id
  --period-col PERIOD_COL
                        the name of the column that contains the light curve
                        periods
  --category-col CATEGORY_COL
                        the name of the column that contains the star category
  --time-col TIME_COL   the name of the column in the curve files that
                        contains the MJD time
  --mag-col MAG_COL     the name of the column in the curve files that
                        contains the magnitude
  --err-col ERR_COL     the name of the column in the curve files that
                        contains the magnitude error
```
