"""
Functions for:

1. Downloading dataset(s) from the Materials Project Database
2. Checking entries in already downloaded dataset(s)
"""

import os
import argparse
import logging

import numpy
import pandas

from pymatgen import MPRester


logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"), format="%(levelname)s:gp-net: %(message)s"
)


def _download(key, opt_property):
    """
    Downloads the dataset (in pickle format) matching the
    optical property of interest.

    :param key: The API key for downloading data
        from the Materials Project database.
    :param opt_property: The optical properties of interest

    :return: None
    """
    api = MPRester(key)
    criteria = {"elements": {"$all": ["O"]}}
    for prop in opt_property:
        print("\nFetching %s data ..." % prop)
        result = api.query(criteria, ["structure", "%s" % prop])

        logging.info("Convert to dataframe ...")
        props_data = pandas.DataFrame(result)
        logging.info("Pickle :)")
        props_data.to_pickle("%s_data.pkl" % prop)


def _read_data(datafile, keep_zeros=False):
    """
    Checks the entries in the pickle dataset so the user can decide
    how to train-test split data for processing.

    :param datafile: The data in .pkl format.
    :param keep_zeros: Include zero optical property values.

    :return: None
    """
    data = pandas.read_pickle(datafile)
    opt_property = datafile.split("/")[-1].split("_data.pkl")[0]
    print(
        "\nNumber of input entries found for %s data = %s" % (opt_property, len(data))
    )
    if keep_zeros:
        logging.info("Including zero optical property values in the dataset...")
        targets = data[opt_property].to_numpy()
        print("Remaining number of entries = %s" % len(targets))
    else:
        logging.info("Excluding zero optical property values from the dataset...")
        mask = numpy.array(
            [i for i, val in enumerate(data[opt_property]) if abs(val) == 0.0]
        )
        numpy.delete(data["structure"].to_numpy(), mask)
        targets = numpy.delete(data[opt_property].to_numpy(), mask)
        print("Remaining number of entries = %s" % len(targets))


def main():
    """From command line, all parsing are handled here"""
    parser = argparse.ArgumentParser(description="MEGNet data download.")
    # Data download options
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MEGNet data in pickle format from Materials Project Database "
        "[default: False]",
        default=False,
    )
    parser.add_argument("--key", help="API key for MEGNet data download", type=str)
    parser.add_argument(
        "--opt_property",
        help="The optical properties of interest separated by spaces when downloading data. "
        "[default: formation_energy_per_atom]",
        default="formation_energy_per_atom",
        type=str,
        nargs="+",
    )

    # Checking entries in already downloaded data
    parser.add_argument(
        "--checkdata",
        action="store_true",
        help="Check number of entries in the dataset. [default: False]",
        default=False,
    )
    parser.add_argument(
        "--data",
        help="Input dataset(s) in pickle formats separated by spaces. ",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--include",
        action="store_true",
        help="Include zero optical property values when checking data entries "
        "[default: False]",
        default=False,
    )
    args = parser.parse_args()

    if args.download:
        # Download data from the Materials Project Database
        _download(args.key, args.opt_property)

    # Check number of entries in dataset
    if args.checkdata:
        for data in args.data:
            _read_data(data, keep_zeros=args.include)


if __name__ == "__main__":
    main()
