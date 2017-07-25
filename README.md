# Light Curve Features

## Usage
```
$ python3 light_curve_features/__main__.py INPUT_FILE CURVES_DIR OUTPUT_FILE --star-id-col ID_COLUMN --period-col PERIOD_COLUMN --category-col CATEGORY_COLUMN
```

Example usage:
```
$ python3 light_curve_features/__main__.py ~/Datasets/eros_2/VariablesInEROS_for_StarDB.csv ~/Datasets/eros_2/curves VariablesInEROS_for_StarDB_extracted.csv --star-id-col EROSID --period-col Period --category-col VARIABLE_CLASS
```
