# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

"""
A script for training a MarconiNet Classifier Model (MCM).
"""

# standard library imports
import configargparse
import os
import json
import urllib.parse
from io import BytesIO

# third party imports
import boto3
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

# local repo imports
from .gpu_scheduler import reserve_gpu_resources, release_gpu_resources
from .marconinet_models import get_marconinet_classifier
from .utils import initialize_tf_gpus, get_logger, set_seed, get_bytes_from_s3_bucket

# comdand line arguments
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

def train(args, logger):
    # model parameters
    logger.info("setting marconinet reconstruction model parameters")
    logger.info(f"number of residual blocks = {len(args.rb_kernsz)}")
    logger.info(f"residual block kernel sizes = {args.rb_kernsz}")
    logger.info(f"residual block dilation rates = {args.rb_dilrate}")

    # load rffp wifi dataset
    logger.info("starting data preparation.")
    logger.info(f"loading data from {args.data_path}")

    if args.data_path.startswith('aws') or args.data_path.startswith('https'):
        data = np.load(BytesIO(get_bytes_from_s3_bucket(args.data_path)))
    else:
        data = np.load(args.data_path)

    x_train = data['x_train']
    y_train = data['y_train']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']
    classes = np.unique(y_train)
    n_classes = len(classes)
    logger.info(f"training dataset size:    {len(y_train)}")
    logger.info(f"validation dataset size:  {len(y_val)}")
    logger.info(f"testing dataset size:     {len(y_test)}")
    logger.info(f"classes: {classes}")
    logger.info(f"n_classes: {n_classes}")

    # reshape data for input into the MCM
    def sigIQ(sigs):
        """normalize & reshape complex-valued time-series."""
        temp = []
        for s in sigs:
            # shift mean to zero
            cpx_mean = np.mean(np.real(s)) + 1j*np.mean(np.imag(s))
            zero_mean = s - cpx_mean
            # normalize magnitude
            mag_norm = zero_mean/np.max(np.abs(s))
            # reshape to (sig_length, 2)
            temp.append(np.array([np.real(mag_norm), np.imag(mag_norm)]).T)
        x = np.array(temp)
        # add dummy dimension necessary for marconinet input
        x = np.expand_dims(x, axis=-1)
        return x

    x_train_mcm = sigIQ(x_train)
    x_val_mcm = sigIQ(x_val)
    x_test_mcm = sigIQ(x_test)
    y_train_ohe = to_categorical(y_train)
    y_val_ohe = to_categorical(y_val)
    y_test_ohe = to_categorical(y_test)

    # Shape information
    input_shape = x_train_mcm.shape[1:]
    output_shape = y_train_ohe.shape[1:]
    logger.info("input shape: {}".format(input_shape))
    logger.info("output shape: {}".format(output_shape))

    # get keras model defined in marconinet_models.py
    architecture_kwargs = {
        "rb_kernel_size": args.rb_kernsz,
        "rb_dilation_rate": args.rb_dilrate,
        "rb_filters": args.rb_filters,
    }
    model = get_marconinet_classifier(input_shape, n_classes, **architecture_kwargs)
    model.summary()

    # compile model
    compile_args = {}
    compile_args["optimizer"] = "adam"
    compile_args["loss"] = "categorical_crossentropy"
    compile_args["metrics"] = ["accuracy"]
    model.compile(**compile_args)

    # Prepare model saving directory and metrics file
    rb_string = "filters_" + str(args.rb_filters)
    rb_string = rb_string + "_ks_" + "_".join(["2"] + [str(i) for i in args.rb_kernsz])
    rb_string = rb_string + "_dr_" + "_".join(["1"] + [str(i) for i in args.rb_dilrate])
    results_id = rb_string + "_seed_{}".format(args.seed)
    if not os.path.isdir(args.metrics_dir):
        os.makedirs(args.metrics_dir)
    if not os.path.isdir(args.save_dir):
        os.makedirs(save_dir)

    if args.model_filename:
        model_name = args.model_filename
    else:
        model_name = f"{model.name}.{results_id}.h5"
    filepath = os.path.join(args.save_dir, model_name)
    metrics_file = f"{model.name}_training_metrics.{results_id}.json"
    metrics_filepath = os.path.join(args.metrics_dir, metrics_file)
    logger.info(f"training metrics file: {metrics_filepath}")

    # setup keras callbacks
    logger.info("setting up keras callbacks")
    chk_monitor = "val_loss"
    lr_monitor = "val_loss"
    estop_monitor = "val_accuracy"
    logger.info(f"model checkpoint tracking {chk_monitor}")
    logger.info(f"reduce learning on plateau tracking {lr_monitor}")
    logger.info(f"early stopping tracking {estop_monitor}")
    callbacks = []
    checkpoints_kwargs = {
        "filepath": filepath,
        "monitor": chk_monitor,
        "verbose": 1,
        "save_best_only": True,
    }
    early_stopping_kwargs = {
        "monitor": estop_monitor,
        "min_delta": 0,
        "patience": args.early_stop_patience,
        "verbose": 1,
        "mode": "auto",
        "baseline": None,
        "restore_best_weights": False,
    }
    reduce_lr_kwargs = {
        "monitor": lr_monitor,
        "factor": 0.2,
        "cooldown": 5,
        "patience": args.reduce_lr_patience,
        "min_lr": 0.5e-6,
        "verbose": 1,
    }
    if checkpoints_kwargs != {}:
        callbacks.append(ModelCheckpoint(**checkpoints_kwargs))
    if early_stopping_kwargs != {}:
        callbacks.append(EarlyStopping(**early_stopping_kwargs))
    if reduce_lr_kwargs != {}:
        callbacks.append(ReduceLROnPlateau(**reduce_lr_kwargs))

    # train model
    logger.info(f"batch_size = {args.batch_size}")
    logger.info(f"training epochs = {args.epochs}")
    logger.info("starting training")
    hist = model.fit(
        x_train_mcm,
        y_train_ohe,
        validation_data=(x_val_mcm, y_val_ohe),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    logger.info("finished training.")

    # save off training metrics
    logger.info("saving training metrics.")
    for k in hist.history.keys():
        # json module doesn't like numpy.float32 objects, cast to regular python floats
        hist.history[k] = [float(metric) for metric in hist.history[k]]
    with open(metrics_filepath, "w") as file:
        file.write(json.dumps(hist.history))

    # test best model
    logger.info("evaluating on test data.")
    bestmodel = keras.models.load_model(filepath)
    test_results = bestmodel.evaluate(
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
    set_seed(args.seed)
    try:
        train(args, logger)
    finally:
        release_gpu_resources(claimed_gpus, status_files)
