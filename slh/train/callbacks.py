import collections
import csv
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger#, TensorBoard # TODO implement tensorboard

from slh.config.common import flatten
from slh.config.train_config import TrainConfig

def format_str(k):
    return f"{k:.5f}"

class CSVLoggerSLH(CSVLogger):
    def __init__(self, filename, separator=",", append=False):
        super().__init__(filename, separator=separator, append=append)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
                return f"\"[{', '.join(map(format_str, k))}]\""
            else:
                return format_str(k)

        if self.keys is None:
            self.keys = sorted(logs.keys())
            # When validation_freq > 1, `val_` keys are not in first epoch logs
            # Add the `val_` keys so that its part of the fieldnames of writer.
            val_keys_found = False
            for key in self.keys:
                if key.startswith("val_"):
                    val_keys_found = True
                    break
            if not val_keys_found:
                self.keys.extend(["val_" + k for k in self.keys])

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs.get(key, "NA"))) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()