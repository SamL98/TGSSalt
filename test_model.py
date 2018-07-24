import util as u

from os.path import join

import numpy as np
from keras.models import model_from_json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gray', dest='gray', action='store_true')
parser.add_argument('-cs', '--cumsum', dest='cs', action='store_true')
parser.add_argument('-n', '--name', dest='name', type=str, default='unet')
parser.add_argument('-t', '--thresh', dest='threshold', type=float, default=-1.0)
args = parser.parse_args()

name = args.name
with open(join('models', name, 'model.json')) as f:
	json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

X, y = u.ips(train=False, gray=args.gray, cs=args.cs)
if len(X.shape) == 3:
	X = np.expand_dims(X, axis=3)

idxs = np.random.choice(range(len(X)), 5)
X = X[idxs]
y_pred = model.predict(X)[:,:,:,0]
y = y[idxs]

for x, true, pred in zip(X, y, y_pred):
	if args.threshold >= 0.0:
		pred = (pred > args.threshold).astype(np.uint8)
	u.disp_imp(x, true, pred)
