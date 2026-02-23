"""
Displays information about the selected layersof a pre-trained MEGNet model
"""

import os
import argparse
import logging

from megnet.models import MEGNetModel

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"), format="%(levelname)s:gp-net: %(message)s"
)


def _show_layers(model_file):
    """
    Displays information about the layers of a pre-trained MEGNet model.

    Inputs:
    :param model_file: A pre-trained MEGNet model file in HDF5 format.

    :return: None
    """
    pretrained_model = MEGNetModel.from_file(model_file)
    print(pretrained_model.summary())


def main():
    """From command line, all parsing are handled here"""
    parser = argparse.ArgumentParser(
        description="Displays layers of MEGNET-trained model file."
    )
    parser.add_argument(
        "--layer",
        help="Display the information about the layer.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    _show_layers(args.ltype)


if __name__ == "__main__":
    main()
