import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf 

from tensorflow.keras import layers
from tensorflow.keras import losses

URL = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

dataset_dir = tf.keras.utils.get_file("stack_overflow_16k", URL,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.dataset
