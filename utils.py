import os
import logging
import numpy as np
import tensorflow as tf

# local repo imports
from .gpu_scheduler import reserve_gpu_resources

def get_logger():
    # setup logging
    logger = logging.getLogger(__name__)
    loglvl = logging.INFO
    logger.setLevel(loglvl)
    consolelog = logging.StreamHandler()
    consolelog.setLevel(loglvl)
    consolelog.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(consolelog)
    return logger

def reset_logging():
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                # Copied from `logging.shutdown`.
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)

def initialize_tf_gpus(vis_gpus, logger):
    logger.debug(f"args.vis_gpus = {vis_gpus}")
    claimed_gpus, status_files = reserve_gpu_resources(numgpus=1, selectfrom=vis_gpus)
    logger.info(f"using gpu(s) {claimed_gpus}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in claimed_gpus)
    return claimed_gpus, status_files

def set_seed(seed, logger):
    # Seed RNGs
    logger.info(f"setting random number generator seed to {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)

