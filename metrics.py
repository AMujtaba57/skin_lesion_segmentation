""" 
This file calculate the intersection over union to find the mask of the image and 
check their Dice loss for evaluating the image with their segmented mask .
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten
from tensorflow.keras.backend import epsilon

gamma = 2
alpha = 0.6


class DiceLoss(tf.keras.losses.Loss):
    """ Calculate the dice loss. """
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = Flatten()(y_true)
        y_pred = Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true*y_pred)
        return 1 - ((2*intersection + epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon()))


def iou(y_true, y_pred):
    """ Calculate the intersection over union. """
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + epsilon()) / (union + epsilon())
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
