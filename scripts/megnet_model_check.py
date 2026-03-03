"""
Displays the layers of a pre-trained MEGNet model
"""

import os
import argparse
import logging

from megnet.models import MEGNetModel

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"), format="%(levelname)s:gp-net: %(message)s"
)


def main():
    """From command line, all parsing are handled here"""
    parser = argparse.ArgumentParser(
        description="Displays layers of MEGNET-trained model file."
    )
    parser.add_argument(
        "--model_file",
        help="The pre-trained MEGNet model file.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    pretrained_model = MEGNetModel.from_file(args.model_file)
    print(pretrained_model.summary())


if __name__ == "__main__":
    main()
