import util as u

import sys
from os.path import join

import numpy as np
from keras.models import model_from_json

name = sys.argv[1]
with open(join('models', name, 'model.json')) as f:
	json = f.read()

model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

X, y = u.ips(train=False)

idxs = np.random.choice(range(len(X)), 5)
X = X[idxs]
y_pred = model.predict(X)[:,:,:,0]
y = y[idxs]

for x, true, pred in zip(X, y, y_pred):
	pred = (pred > 0.5).astype(np.uint8)
	u.disp_imp(x, true, pred)
