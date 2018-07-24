import util as u

from os.path import join

u.silence()
import numpy as np
from keras.models import model_from_json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, default='unet')
parser.add_argument('-t', '--thresh', dest='threshold', type=float, default=-1.0)
args = parser.parse_args()

name = args.name
with open(join('models', name, 'model.json')) as f:
	json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

num_chan = model.layers[0].input_shape[-1]
if num_chan == 3:
	u.set_gray(False)
	u.set_cs(False)
elif num_chan == 2:
	u.set_cs(True)
else:
	u.set_cs(False)
	u.set_gray(True)

X, y = u.ips(train=False)

u.unsilence()
print(X.shape, y.shape)

if len(X.shape) == 3:
	X = np.expand_dims(X, axis=3)

idxs = np.random.choice(range(len(X)), 5, replace=False)
X = X[idxs]
y = y[idxs]
y_pred = model.predict(X)[:,:,:,0]

if (X[0].shape)[2] == 2:
	X = X[:,:,:,0]

for x, true, pred in zip(X, y, y_pred):
	if args.threshold >= 0.0:
		pred = (pred > args.threshold).astype(np.uint8)
	u.disp_imp(x, true, pred)
