import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.backend import epsilon


y_pred = cv2.imread('predicted_img/ISIC_0000214.jpg')
y_true = cv2.imread('image/ISIC_0000214.jpg') 

y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2GRAY)
y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2GRAY)



y_true = cv2.resize(y_true, (224, 224))
y_pred = cv2.resize(y_pred, (224, 224))


def dice(pred, true, k = 1):
    intersection = np.sum(pred[true!=k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


dice_score = dice_coef(y_pred, y_true) 
print ("Dice Similarity Coefficient: {}".format(dice_score))

dice_score = dice(y_pred, y_true, k=255) 
print ("Dice Similarity Score: {}".format(dice_score))

