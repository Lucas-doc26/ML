import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from functools import partial
import albumentations as A

AUTOTUNE = tf.data.experimental.AUTOTUNE

data, info= tfds.load(name="tf_flowers", split="train", as_supervised=True, with_info=True)
data

info
