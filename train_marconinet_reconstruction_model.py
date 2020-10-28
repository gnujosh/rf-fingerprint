# -----------------------------------------------------------------------------------
# ---------------------------------- UNCLASSIFIED -----------------------------------
# This code was developed under the Defense Advanced Research Projects Agency (DARPA)
# Radio Frequency Machine Learning System (RFMLS) program: contract FA8750-18-C-0150.
# Government Unlimited Rights
# BAE Systems, Inc. (C) 2020.
# -----------------------------------------------------------------------------------

"""
A script for training and testing a MarconiNet Reconstruction Model (MRM).
"""

# standard library imports
import configargparse
import os
import random
import logging
from datetime import datetime
import json

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    EarlyStopping,
)

# local repo imports
from gpu_scheduler import reserve_gpu_resources
from gpu_scheduler import release_gpu_resources
from digital_wifi_dataset import WifiData
from digital_wifi_dataset import WifiReconstruct
from marconinet_models import get_marconinet_reconstruction_model

# comdand line arguments
def parseArgs():
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
        default=300,
        type=int,
        help="number of training epochs",
    )
    parser.add_argument(
        "--rb_dilrate",
        default=[1, 4, 16, 64, 192, 256],
        type=int,
        help="residual block dilation rates",
        nargs="*",
    )
    parser.add_argument(
        "--rb_kernsz",
        default=[4, 4, 4, 4, 4, 4],
        type=int,
        help="residual block kernel sizes",
        nargs="*",
    )
    parser.add_argument(
        "--rb_filters",
        default=25,
        type=int,
        help="number of filters per residual block",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=50,
        type=int,
        help="number of epochs with no improvement before stopping training",
    )
    parser.add_argument(
        "--reduce_lr_patience",
        default=30,
        type=int,
        help="number of epochs with no improvement before reducing the lr",
    )
    parser.add_argument(
        "plot_training_metrics",
        action="store_true",
        help="launch browser to show training metrics",
    )
    return parser.parse_args()

# read config and parse commandline arguments
args = parseArgs()

# setup logging
logger = logging.getLogger(__name__)
loglvl = logging.INFO
logger.setLevel(loglvl)
consolelog = logging.StreamHandler()
consolelog.setLevel(loglvl)
consolelog.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(consolelog)

# Seed RNGs
logger.info(f"setting random number generator seed to {args.seed}")
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# reserve gpu resources if running on a mabnunxlssep server. You can comment this section
# of code out if you are not running on a mabnunxlssep server.
logger.debug(f"args.vis_gpus = {args.vis_gpus}")
claimedgpus, statusFiles = reserve_gpu_resources(numgpus=1, selectfrom=args.vis_gpus)
logger.info(f"using gpu(s) {claimedgpus}")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in claimedgpus)

def main():
    # model parameters
    logger.info("setting marconinet reconstruction model parameters")
    logger.info(f"number of residual blocks = {len(args.rb_kernsz)}")
    logger.info(f"residual block kernel sizes = {args.rb_kernsz}")
    logger.info(f"residual block dilation rates = {args.rb_dilrate}")

    # define data transformation
    def sigIQ_shift(sig):
        """Augment with random translation then reshape complex valued data to
        [Real,Complex]."""
        # randomely shift/translate signal in time
        shift = np.random.randint(0, len(sig))
        sig = np.roll(sig, shift)
        x = np.array([np.real(sig), np.imag(sig)]).T
        return np.expand_dims(x, axis=-1)

    # setup wifi data keras generators
    logger.info("starting data preparation.")
    wifigen = WifiData("digital_wifi_dataset_files")
    train = WifiReconstruct(
        wifigen,
        batch_size=args.batch_size,
        slice_start=0,
        slice_end=0.75,
        transformation=sigIQ_shift,
    )
    val = WifiReconstruct(
        wifigen,
        batch_size=args.batch_size,
        slice_start=0.75,
        slice_end=0.9,
        transformation=sigIQ_shift,
    )
    test = WifiReconstruct(
        wifigen,
        batch_size=args.batch_size,
        slice_start=0.9,
        slice_end=1,
        transformation=sigIQ_shift,
    )

    # Shape information
    x, y = train[0]
    xv, yv = val[0]
    input_shape = x.shape[1:]
    output_shape = y.shape[1:]
    logger.info("input shape: {}".format(input_shape))
    logger.info("output shape: {}".format(output_shape))

    # get keras model defined in marconinet_models.py
    architecture_kwargs = {
        "rb_kernel_size": args.rb_kernsz,
        "rb_dilation_rate": args.rb_dilrate,
        "rb_filters": args.rb_filters,
    }
    model = get_marconinet_reconstruction_model(input_shape, **architecture_kwargs)
    model.summary()

    # compile model
    compile_args = {}
    compile_args["optimizer"] = "adam"
    compile_args["loss"] = {"reconstruction_layer": "mean_squared_error"}
    compile_args["metrics"] = {"reconstruction_layer": "mean_squared_error"}
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
    model_name = f"{model.name}.{results_id}.h5"
    filepath = os.path.join(args.save_dir, model_name)
    metrics_file = f"{model.name}_training_metrics.{results_id}.json"
    metrics_filepath = os.path.join(args.metrics_dir, metrics_file)
    logger.info(f"training metrics file: {metrics_filepath}")

    # setup keras callbacks
    logger.info("setting up keras callbacks")
    accuracy_to_mointor = "val_loss"
    logger.info(f"early stopping and lr tracking {accuracy_to_mointor}")
    callbacks = []
    checkpoints_kwargs = {
        "filepath": filepath,
        "monitor": accuracy_to_mointor,
        "verbose": 1,
        "save_best_only": True,
    }
    early_stopping_kwargs = {
        "monitor": accuracy_to_mointor,
        "min_delta": 0,
        "patience": args.early_stop_patience,
        "verbose": 1,
        "mode": "auto",
        "baseline": None,
        "restore_best_weights": False,
    }
    reduce_lr_kwargs = {
        "monitor": accuracy_to_mointor,
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
    hist = model.fit_generator(
        train,
        epochs=args.epochs,
        validation_data=val,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    )
    logger.info("finished training.")

    # save off training metrics
    logger.info("saving trianing metrics.")
    for k in hist.history.keys():
        # json module doesn't like numpy.float32 objects, cast to regular python floats
        hist.history[k] = [float(metric) for metric in hist.history[k]]
    with open(metrics_filepath, "w") as file:
        file.write(json.dumps(hist.history))

    # test best model
    logger.info("evaluating on test data.")
    bestmodel = keras.models.load_model(filepath)
    test_results = bestmodel.evaluate_generator(test)
    logger.info(f"test results: {str(test_results)}")

    # plot training metrics
    from bokeh.plotting import figure, output_file, show
    from bokeh.layouts import row
    output_file(f"{model.name}_training_metrics_plots.{results_id}.html")
    train_color = "blue"
    val_color = "orange"
    epochs = np.arange(1,len(hist.history["loss"])+1)
    tloss = hist.history["loss"]
    vloss = hist.history["val_loss"]
    lr = hist.history["lr"]
    # loss plot
    ploss = figure(title="Loss")
    ploss.line(epochs, tloss, line_color=train_color, legend_label="train")
    ploss.line(epochs, vloss, line_color=val_color, legend_label="val")
    # learning rate plot
    plr = figure(title="Learning Rate")
    plr.line(epochs, lr, line_color=train_color)
    r = row(ploss, plr)
    if args.plot_training_metrics:
        show(r)

# launch main()
if __name__ == "__main__":
    try:
        main()
    finally:
        release_gpu_resources(claimedgpus, statusFiles)
