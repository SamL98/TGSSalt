import util as u
from iou import iou_metric

from os.path import join

#u.silence()
import numpy as np
from keras.models import model_from_json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', type=str, default='unet')
parser.add_argument('-no', '--num', dest='num', type=int, default=5)
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

ids = u.img_ids(train=True)
ids = np.random.choice(ids, args.num, replace=False)

X, y = u.ips(for_ids=ids)
if len(X.shape) == 3:
	X = np.expand_dims(X, axis=3)

u.unsilence()
print(X.shape, y.shape)

y_pred = model.predict(X)[:,:,:,0]

if (X[0].shape)[2] == 2 or (X[0].shape)[2] == 1:
	X = X[:,:,:,0]

for x, true, pred in zip(X, y, y_pred):
	if args.threshold >= 0.0:
		pred = (pred > args.threshold).astype(np.uint8)
	
	iou = iou_metric(true, pred)
	print('IoU: %f' % iou)

	u.disp_imp(x, true, pred)
