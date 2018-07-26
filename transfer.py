import util as u
import iou
from logger import Logger

import os
from os.path import join, isdir

#u.silence()
from keras.applications import VGG19
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy as bce

import numpy as np
import tensorflow as tf

import argparse
def get_args():
	parser = argparse.ArgumentParser('U-Net for TGS Salt Identification')
	parser.add_argument('-n', '--name', dest='name', type=str, default='unet')
	parser.add_argument('-dr', '--dropout', dest='dropout', type=float, default=0.0)
	parser.add_argument('-nm', '--norm', dest='normalize', action='store_true')
	parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=50)
	parser.add_argument('-di', '--dice', dest='dice', action='store_true')
	parser.add_argument('-t', '--test', dest='test', action='store_true')
	parser.add_argument('-f', '--flip', dest='flip', action='store_true')
	parser.add_argument('-de', '--denoise', dest='denoise', action='store_true')
	parser.add_argument('-nf', '--num_freeze', dest='num_freeze', type=int, default=5)
	return parser.parse_args()

args = get_args()

# Load the data and add an axis to the end of y
# so that it is three-dimensional
u.set_gray(False)
X, y = u.ips()
y = np.expand_dims(y, axis=3)

if args.test:
	X, y = X[:100], y[:100]

if args.flip:
	X = np.append(X, [np.fliplr(x) for x in X], axis=0)
	y = np.append(y, [np.fliplr(y) for y in y], axis=0)

if len(X.shape) == 3:
	X = np.expand_dims(X, axis=3)

u.unsilence()
print(X.shape, y.shape)

name = args.name # name of the model
if not isdir(join('logs', name)):
	os.mkdir(join('logs', name))

dropout = args.dropout # dropout to use in the U-Net

# Construct the U-Net
def get_model():
	vgg = VGG19(weights='imagenet', include_top=False, input_shape=X[0].shape)

	for layer in vgg.layers[:args.num_freeze]:
		layer.trainable = False

	inputs = vgg.output

	u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (inputs)
	u6 = concatenate([u6, vgg.layers[-2].output])
	c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (u6)
	c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (c6)
	c6 = Conv2D(512, (3, 3), activation='relu', padding='same') (c6)
	d6 = Dropout(dropout) (c6)

	u7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (d6)
	u7 = concatenate([u7, vgg.layers[-7].output])
	c7 = Conv2D(512, (3, 3), activation='relu', padding='same') (u7)
	c7 = Conv2D(512, (3, 3), activation='relu', padding='same') (c7)
	c7 = Conv2D(512, (3, 3), activation='relu', padding='same') (c7)
	d7 = Dropout(dropout) (c7)

	u8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (d7)
	u8 = concatenate([u8, vgg.layers[-12].output])
	c8 = Conv2D(256, (3, 3), activation='relu', padding='same') (u8)
	c8 = Conv2D(256, (3, 3), activation='relu', padding='same') (c8)
	c8 = Conv2D(256, (3, 3), activation='relu', padding='same') (c8)

	u9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c8)
	u9 = concatenate([u9, vgg.layers[-17].output], axis=3)
	c9 = Conv2D(128, (3, 3), activation='relu', padding='same') (u9)
	c9 = Conv2D(128, (3, 3), activation='relu', padding='same') (c9)

	u10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c9)
	u10 = concatenate([u10, vgg.layers[-20].output], axis=3)
	c10 = Conv2D(64, (3, 3), activation='relu', padding='same') (u10)
	c10 = Conv2D(64, (3, 3), activation='relu', padding='same') (c10)

	outputs = Conv2D(1, (1, 1), activation='sigmoid') (c10)
	model = Model(inputs=[vgg.input], outputs=[outputs])
	return model

model = get_model()

"""
Custom Keras metric for evaluating the IoU
:param y_true: batch of ground truth masks
:param y_pred: batch of predicted masks
"""
def mean_iou(y_true, y_pred):
	return tf.py_func(iou.batch_iou, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred, smooth=1):
	intersection = K.sum(y_true * y_pred, axis=[1,2,3])
	union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
	return K.mean((2.*intersection + smooth) / (union + smooth), axis=0)

def bce_dice(y_true, y_pred):
	return 0.01*bce(y_true, y_pred) - dice_coef(y_true, y_pred)

def bce_nodice(y_true, y_pred):
	return bce(y_true, y_pred)

def true_postive_rate(y_true, y_pred):
	return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred)))/K.sum(y_true)

# Callbacks for early stopping and saving
# model weights if the mean IoU on the validation set improves
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1)
early_stop = EarlyStopping(patience=5, verbose=1)
checkpoint = ModelCheckpoint(
	join('models', name, 'model.h5'),
	monitor='val_mean_iou', mode='max',
	verbose=1, save_best_only=True)

loss = bce_nodice
if args.dice:
	loss = bce_dice

# Compile the model using binary_crossentropy for the loss function
# and adam as the optimizer.
model.compile(
	loss=loss,
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
	callbacks=[
		early_stop, checkpoint, reduce_lr, 
		Logger(join('logs', name, 'metrics.csv'))])
