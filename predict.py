"""
This file perform predictions over 60-coloured images which are given in the "image" folder.
"""

import tensorflow as tf
import numpy as np
from data import get_predict
import cv2
import os
import sklearn.preprocessing
file_path_model = 'pre-trained_models/model.h5'
trained_model = tf.keras.models.load_model(file_path_model, compile=False)
size = (512, 384)
images_folder = 'image'
file_img = os.listdir(images_folder)

abc = []
for index, i in enumerate(file_img):
    name = images_folder + '/'+i
    img2 = cv2.imread(name, cv2.IMREAD_COLOR)
    out_put = get_predict(trained_model, img2)
    out_put = out_put.astype(np.uint8)
    out_put = np.stack([out_put, out_put, out_put], axis=-1)
    abc.append(out_put)
    cv2.imwrite(f"predicted_img/{i}", out_put)
