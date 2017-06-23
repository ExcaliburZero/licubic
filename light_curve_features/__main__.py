from os import path

import argparse
import pandas as pd
import sys

import feature_extract

def main():
    """
    $ python3 __main__.py CatalinaVars.csv curves/ test.csv
    """
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    result = validate_arguments(args)

    if isinstance(result, str):
        print(result, file=sys.stderr)
        sys.exit(1)

    data = pd.read_csv(args.data_file, nrows=args.nrows)

    new_data = feature_extract.process_data(
            data,
            args.curves_dir,
            save_curve_files=args.save_curve_files
        )

    new_data.to_csv(args.output_file, index=False)

def create_arg_parser():
    """
    Creates and returns the command line arguments parser for the script.

    Returns
    -------
    arg_parser : argparse.ArgumentParser
        The command line argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Extract features from CRTS light curve data.")

    # Required arguments
    parser.add_argument("data_file", type=str,
            help="the input data file")
    parser.add_argument("curves_dir", type=str,
            help="the directory where the light curves are stored")
    parser.add_argument("output_file", type=str,
            help="the output data file")

    # Optional flags
    parser.add_argument("--nrows", dest="nrows", type=int, default=None,
            help="the number of rows of data to process (default: all)")
    parser.add_argument("--save-curves", dest="save_curve_files", action="store_const",
            const=True, default=False,
            help="save the intermediate light curves")

    return parser

def validate_arguments(args):
    """
    Checks to see if the given command line arguments are valid. Returns None
    if they are all valid, or an error string if one or more are invalid.

    Parameters
    ---------
    args : argparse.Namespace
        The parsed command line arguments.

    Returns
    -------
    error : Union[None, str]
        The error message if at least one argument was invalid, otherwise is
        None.
    """
    data_file = args.data_file
    curves_dir = args.curves_dir
    nrows = args.nrows

    if not path.exists(data_file):
        return "The given data file does not exist: %s" % data_file
    if not path.isfile(data_file):
        return "The given data file is not a file: %s" % data_file

    if not path.exists(curves_dir):
        return "The given curve file directory does not exist: %s" % curves_dir
    if not path.isdir(curves_dir):
        return "The given curve file directory is not a directory: %s" % curves_dir

    if nrows is not None:
        if nrows < 0:
            return "The given nrows is not a non-negative integer: %s" % nrows

    return None

if __name__ == "__main__":
    main()
