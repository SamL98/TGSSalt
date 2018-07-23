import util as u

import os
from os.path import join
import sys

import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.morphology import label
from keras.models import model_from_json

name = sys.argv[1]

with open(join('models', name, 'model.json')) as f:
	json = f.read()
model = model_from_json(json)
model.load_weights(join('models', name, 'model.h5'))

"""
Perform RLE on a connected component of a mask
:param x: the image to encode
"""
def rle_encoding_for_comp(x):
	# I stole this code from Kaggle and have not put in the effort
	# to learn how it works.
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

"""
Perform RLE encoding on an image
:param x: the image to encode
"""
def rle_encoding(x):
	x = label(x)
	return [rle_encoding_for_comp(x == i) for i in range(1, x.max()+1)]

# load the unlabeled test images
X = u.test_imgs()
ids = u.test_ids()
ids = [i[:i.index('.')] for i in ids]

# propagate the test images forward through the U-Net
pred = model.predict(X, batch_size=64)[:,:,:,0]

# the original dimension of the images
odim = 101

rle_ids = []
rles = []

for i, p in enumerate(pred):
	if i % 100 == 0:
		print('%d/%d' % (i, len(X)))

	p = resize(p, (odim, odim), mode='constant', preserve_range=True)
	mask = (p > 0.35).astype(np.uint8)
	mask_rle = rle_encoding(mask)
	
	rles.extend(mask_rle)
	rle_ids.extend([ids[i]] * len(mask_rle))

	if len(mask_rle) == 0:
		rles.extend([[1, 1]])
		rle_ids.extend([ids[i]])	

df = pd.DataFrame()
df['id'] = rle_ids
df['rle_mask'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
df.to_csv(join('models', name, 'sub.csv'), index=False)
