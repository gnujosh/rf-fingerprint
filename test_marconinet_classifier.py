# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

"""
A script for testing a MarconiNet Classifier Model (MCM).
"""

# standard library imports
import configargparse
import os

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical

# local repo imports
from .gpu_scheduler import reserve_gpu_resources, release_gpu_resources
from .utils import initialize_tf_gpus, get_logger, set_seed


# command line arguments
def parse_args(arguments=None):
    """Get commandline options."""
    parser = configargparse.ArgParser(
        description=__doc__,
        formatter_class=configargparse.ArgumentDefaultsRawHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        is_config_file=True,
        help="path to config file. cmdline args override config file options.",
    )
    parser.add_argument(
        "--save_dir", default=os.getcwd(), type=str, help="where to save model"
    )
    parser.add_argument(
        "--metrics_dir",
        default=os.getcwd(),
        type=str,
        help="where to save training metrics",
    )
    parser.add_argument(
        "--vis_gpus",
        default=[str(i) for i in range(8)],
        nargs="*",
        help="pool of gpus to select from",
    )
    parser.add_argument(
        "--seed", default=1337, type=int, help="random number generator seed"
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="training batch size",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="max number of training epochs",
    )
    parser.add_argument(
        "--rb_dilrate",
        default=[2, 4, 8, 16, 32, 64, 128, 256],
        type=int,
        help="residual block dilation rates",
        nargs="*",
    )
    parser.add_argument(
        "--rb_kernsz",
        default=8*[4],
        type=int,
        help="residual block kernel sizes",
        nargs="*",
    )
    parser.add_argument(
        "--rb_filters",
        default=50,
        type=int,
        help="number of filters per residual block",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=5,
        type=int,
        help="number of epochs with no improvement before stopping training",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        default=3,
        type=int,
        help="number of epochs with no improvement before reducing the lr",
    )
    parser.add_argument(
        "--plot_training_metrics",
        action="store_true",
        help="launch browser to show training metrics",
    )
    parser.add_argument(
        "--data_path",
        help="Path to npz data file"
    )
    parser.add_argument(
        "--model_filename",
        help="Filename of model file"
    )
    if arguments is None:
        return parser.parse_args()
    else:
        return parser.parse_args(arguments)

def test(args, logger):

    # load rffp wifi dataset
    logger.info("starting data preparation.")
    logger.info(f"loading data from {args.data_path}")
    data = np.load(args.data_path)
    x_test = data['x_test']
    y_test = data['y_test']
    logger.info(f"testing dataset size:     {len(y_test)}")

    # reshape data for input into the MCM
    def sigIQ(sigs):
        """normalize & reshape complex-valued time-series."""
        temp = []
        for s in sigs:
            # shift mean to zero
            cpx_mean = np.mean(np.real(s)) + 1j * np.mean(np.imag(s))
            zero_mean = s - cpx_mean
            # normalize magnitude
            mag_norm = zero_mean / np.max(np.abs(s))
            # reshape to (sig_length, 2)
            temp.append(np.array([np.real(mag_norm), np.imag(mag_norm)]).T)
        x = np.array(temp)
        # add dummy dimension necessary for marconinet input
        x = np.expand_dims(x, axis=-1)
        return x

    x_test_mcm = sigIQ(x_test)
    y_test_ohe = to_categorical(y_test)

    model = keras.models.load_model(os.path.join(args.save_dir, args.model_filename))

    # test best model
    logger.info("evaluating on test data.")
    test_results = model.evaluate(
        x_test_mcm,
        y_test_ohe,
        batch_size=args.batch_size,
    )
    logger.info(f"test results: {str(test_results)}")


if __name__ == "__main__":

    args = parse_args()
    logger = get_logger()

    # reserve gpu resources if running on a mabnunxlssep server. You can comment this section
    # of code out if you are not running on a mabnunxlssep server.
    claimed_gpus, status_files = initialize_tf_gpus(args.vis_gpus, logger)
    set_seed(args.seed, logger)
    try:
        test(args, logger)
    finally:
        release_gpu_resources(claimed_gpus, status_files)