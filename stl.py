import util as u
import iou
from logger import Logger

import os
from os.path import join, isdir

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, model_from_json

import numpy as np
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser('U-Net for TGS Salt Identification')
parser.add_argument('-n', '--name', dest='name', type=str, default='u-net')
parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.0)
parser.add_argument('-an', '--ae_name', dest='ae_name', type=str, default='sae')
parser.add_argument('-t', '--train', dest='trainable', action='store_true')
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=25)
args = parser.parse_args()

with open(join('models', args.ae_name, 'model.json')) as f:
	json = f.read()
ae = model_from_json(json)
ae.load_weights(join('models', args.ae_name, 'model.h5'))

last_layer = ae.layers[-1]
while not last_layer.name == "conv2d_10":
	ae.layers.pop()
	last_layer = ae.layers[-1]

for layer in ae.layers:
	layer.trainable = args.trainable

num_chan = ae.layers[0].input_shape[-1]
if num_chan == 3:
	u.set_gray(False)
	u.set_cs(False)
elif num_chan == 2:
	u.set_cs(True)
else:
	u.set_cs(False)
	u.set_gray(True)

# Load the data and add an axis to the end of y
# so that it is three-dimensional
X, y = u.ips()
y = np.expand_dims(y, axis=3)

name = args.name # name of the model
dropout = args.dropout # dropout to use in the U-Net

i = -5
u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='u6') (ae.layers[-1].output)
u6 = concatenate([u6, ae.layers[i].output])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='c6-1') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='c6-2') (c6)
d6 = Dropout(dropout, name='d6') (c6)

i -= 4
u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='u7') (d6)
u7 = concatenate([u7, ae.layers[i].output])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='c7-1') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='c7-2') (c7)
d7 = Dropout(dropout, name='d7') (c7)

i -= 3
u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='u8') (d7)
u8 = concatenate([u8, ae.layers[i].output])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='c8-1') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='c8-2') (c8)

i -= 3
u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', name='u9') (c8)
u9 = concatenate([u9, ae.layers[i].output], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='c9-1') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='c9-2') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid', name='out') (c9)
model = Model(inputs=[ae.layers[0].input], outputs=[outputs])

"""
Custom Keras metric for evaluating the IoU
:param y_true: batch of ground truth masks
:param y_pred: batch of predicted masks
"""
def mean_iou(y_true, y_pred):
	return tf.py_func(iou.batch_iou, [y_true, y_pred], tf.float32)

# Callbacks for early stopping and saving
# model weights if the mean IoU on the validation set improves
early_stop = EarlyStopping(patience=5, verbose=1)
checkpoint = ModelCheckpoint(
	join('models', name, 'model.h5'),
	monitor='val_mean_iou', mode='max',
	verbose=1, save_best_only=True)

# Compile the model using binary_crossentropy for the loss function
# and adam as the optimizer.
model.compile(
	loss='binary_crossentropy',
	optimizer='adam',
	metrics=[mean_iou])

# Create a directory for the current model
# sys.argv[1] should be the name of the model being trained
if not isdir(join('models', name)):
	os.mkdir(join('models', name))

# Save the model architecture to JSON
with open(join('models', name, 'model.json'), 'w') as f:
	f.write(model.to_json())

# Train the model for 25 epochs using a batch size of 64
model.fit(
	X, y,
	epochs=args.epochs, batch_size=64,
	validation_split=0.1, shuffle=True,
	callbacks=[early_stop, checkpoint, Logger(join('logs', name, 'metrics.csv'))])
