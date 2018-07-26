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
#X = np.array([u.preprocess(x) for x in X])

if len(X.shape) == 3:
	X = np.expand_dims(X, axis=3)

u.unsilence()

y_pred = model.predict(X)[:,:,:,0]

if (X[0].shape)[2] == 2 or (X[0].shape)[2] == 1:
	X = X[:,:,:,0]

from skimage.morphology import label
from skimage.measure import find_contours
import cv2 as cv

def create_diff(x):
	diff = get_diffs(x, rev=False)
	diff = np.maximum(diff, get_diffs(x, rev=True))
	return diff

def get_diffs(img, rev=False, zval=0):
	zrow = np.zeros((1, img.shape[1]), dtype=np.uint8)+zval
	zcol = np.zeros((img.shape[0], 1), dtype=np.uint8)+zval

	if not rev:
		shifted = np.concatenate((img[1:], zrow), axis=0)
		shifted = np.concatenate((shifted[:,1:], zcol), axis=1)
	else:
		shifted = np.concatenate((zrow, img[1:]), axis=0)
		shifted = np.concatenate((zcol, shifted[:,1:]), axis=1)

	return shifted-img

'''
def fill_corners(mask):
	w = 7

	if np.count_nonzero(mask[0:w, 0:w]) > '''

def postprocessing(mask):
	comps = label(mask)

	for i in range(1, len(np.unique(comps))):
		if np.sum(comps==i) < 300:
			mask[comps==i] = 0

	for i in range(5):
		mask = cv.medianBlur(mask, 5)

	contours = find_contours(mask, 0.0)
	#print(contours)

	return mask

import matplotlib.pyplot as plt

for x, true, pred in zip(X, y, y_pred):
	b = (128-101)//2
	x = x[b:-b, b:-b]
	x = u.preprocess(x)

	true = true[b:-b, b:-b]
	pred = pred[b:-b, b:-b]

	pred = (pred > 0.5).astype(np.uint8)
	iou_reg = iou_metric(true, pred)

	post = postprocessing(pred)
	iou_post = iou_metric(true, post)

	#print('IoU: %f' % (iou))
	#u.disp_2x2([x, true, low_pred, hi_pred], ['Input', 'Truth', 'IoU: %f' % iou_prev, 'IoU: %f' % iou])
	
	_, ax = plt.subplots(2, 2)
	ax[0,0].imshow(x, cmap='gray')
	ax[0,0].set_title('Input')
	ax[0,0].axis('off')

	ax[0,1].imshow(x, cmap='gray')
	ax[0,1].imshow(true, alpha=0.3, cmap='OrRd')
	ax[0,1].set_title('True')
	ax[0,1].axis('off')

	ax[1,0].imshow(x, cmap='gray')
	ax[1,0].imshow(pred, alpha=0.3, cmap='Greens')
	ax[1,0].set_title('Pred: %f' % iou_reg)
	ax[1,0].axis('off')

	ax[1,1].imshow(x, cmap='gray')
	ax[1,1].imshow(post, alpha=0.3, cmap='Greens')
	ax[1,1].set_title('Pred (post): %f' % iou_post)
	ax[1,1].axis('off')

	plt.show()