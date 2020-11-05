import os
import logging
import urllib.parse
import contextlib

import boto3
import h5py
import numpy as np
import tensorflow as tf

# local repo imports
from gpu_scheduler import reserve_gpu_resources

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

def get_bytes_from_s3_bucket(filename):
    # Get body bytes of object in s3 bucket. Supports formats like:
    # aws://{aws_access_key_id}:{aws_secret_access_key}@{region_name}/{bucket_name}/{model_file_name}
    # https://{aws_access_key_id}:{aws_secret_access_key}@{bucket_name}.s3-{region_name}.amazonaws.com/{module_file_name}

    url = urllib.parse.urlparse(filename)
    if filename.startswith('aws'):
        aws_access_key_id, aws_secret_access_key, region_name = url.username, url.password, url.hostname
        split_ind = url.path[1:].find('/') + 1
        bucket_name, key_name = url.path[1:split_ind], url.path[split_ind+1:]
    elif filename.startswith('https'):
        aws_access_key_id, aws_secret_access_key, key_name = url.username, url.password, url.path[1:]
        split_ind = url.hostname.find('.')
        split_ind2 = url.hostname.find('.', split_ind + 1)
        bucket_name = url.hostname[:split_ind]
        region_name = url.hostname[split_ind + 4:split_ind2]

    # Load object from bucket
    session = boto3.Session(aws_access_key_id=aws_access_key_id,
                            aws_secret_access_key=aws_secret_access_key,
                            region_name=region_name)

    obj = session.client('s3').get_object(Bucket=bucket_name, Key=key_name)
    return obj['Body'].read()

def load_model_from_s3_bucket(filename):
    body = get_bytes_from_s3_bucket(filename)
    # Fancy way to load data into a memory-backed h5 file instead of copying 
    # file from bucket to local storage.
    file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    file_access_property_list.set_fapl_core(backing_store=False)
    file_access_property_list.set_file_image(body)

    file_id_args = {'fapl': file_access_property_list, 'flags': h5py.h5f.ACC_RDONLY, 'name': b'tmp'} # 'name' doesn't matter
    h5_file_args = {'backing_store': False, 'driver': 'core', 'mode': 'r'}

    with contextlib.closing(h5py.h5f.open(**file_id_args)) as file_id:
        with h5py.File(file_id, **h5_file_args) as h5_file:
            model = tf.keras.models.load_model(h5_file)

    return model