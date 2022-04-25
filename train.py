"""
This is the training file that trained our model over skin-lesion image dataset.
This file  read all the coloured and segmented images and list them into training, validation
and test dataset. 
"""

from utils import transform_name_for_x_train, transform_name_for_y_train
from model import create_double_u_net
import tensorflow as tf
import numpy as np
import os
from data import data_squence
from metrics import DiceLoss, iou
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import warnings
import pathlib
warnings.filterwarnings('ignore')

filename = os.listdir('validation/mask')
x_test = list(map(lambda a: 'validation/image/' +
                  a.replace('_segmentation.png', '.jpg'), filename))
y_test = list(map(lambda a: 'validation/mask/'+a, filename))

file_name = os.listdir('data_augmented/mask')
x_train = list(map(transform_name_for_x_train, file_name))
y_train = list(map(transform_name_for_y_train, file_name))

train = data_squence(x_train, y_train, batch_size=16,)
validation = data_squence(x_test, y_test, batch_size=16, )

pathlib.Path("./checkpoint").mkdir(exist_ok=True, parents=True)

mc = ModelCheckpoint(filepath=os.path.join(
    './checkpoint', "model_ep{epoch:03d}.h5"), save_weights_only=False, save_format="h5", verbose=1)
tb = TensorBoard(log_dir='./log', write_graph=True)

model = create_double_u_net((192, 256, 3))

loss_function = DiceLoss()
model.compile('Adam', loss=loss_function, metrics=[iou])
model.fit(train, epochs=100, validation_data=validation,
          initial_epoch=0,
          verbose=1,
          callbacks=[tb, mc])

