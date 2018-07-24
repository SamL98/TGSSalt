import util as u
import iou

import os
from os.path import join, isdir

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model

import numpy as np
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser('U-Net for TGS Salt Identification')
parser.add_argument('-n', '--name', dest='name', type=str, default='u-net')
parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.0)
parser.add_argument('-cs', '--cumsum', dest='cs', action='store_true')
parser.add_argument('-g', '--gray', dest='gray', action='store_true')
parser.add_argument('-nm', '--norm', dest='normalize', action='store_true')
args = parser.parse_args()

u.set_gray(args.gray)
u.set_cs(args.cs)

# Load the data and add an axis to the end of y
# so that it is three-dimensional
X, y = u.ips()
y = np.expand_dims(y, axis=3)
print(X.shape, y.shape)

name = args.name # name of the model
dropout = args.dropout # dropout to use in the U-Net

# Construct the U-Net
inputs = Input(X[0].shape)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)
d3 = Dropout(dropout) (p3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (d3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
d4 = Dropout(dropout) (p4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (d4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)
d6 = Dropout(dropout) (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (d6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)
d7 = Dropout(dropout) (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (d7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
model = Model(inputs=[inputs], outputs=[outputs])

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
	epochs=25, batch_size=64,
	validation_split=0.1, shuffle=True,
	callbacks=[early_stop, checkpoint])
