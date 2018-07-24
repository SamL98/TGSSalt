import util as u

import os
from os.path import join

import numpy as np
import pandas as pd

from skimage.transform import resize
from skimage.morphology import label
from keras.models import model_from_json

import argparse
parser = argparse.ArgumentParser('U-Net for TGS Salt Identification')
parser.add_argument('-n', '--name', dest='name', type=str, default='u-net')
parser.add_argument('-t', '--threshold', dest='tval', type=float, default=0.5)
parser.add_argument('-cs', '--cumsum', dest='cs', action='store_true')
parser.add_argument('-g', '--gray', dest='gray', action='store_true')
parser.add_argument('-sb', '--sub_name', dest='sub_name', type=str, default='sub')
args = parser.parse_args()

u.set_gray(args.gray)
u.set_cs(args.cs)
name = args.name

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
	return [rle_encoding_for_comp(x)]

odim = 101 # the original dimension of the images

rle_ids = []
rles = []

def encode_batch(pred, ids):
	global rle_ids, rles, odim
	for i, p in enumerate(pred):
		p = resize(p, (odim, odim), mode='constant', preserve_range=True)
		mask = (p > args.tval).astype(np.uint8)
		mask_rle = rle_encoding(mask)
		
		rles.extend(mask_rle)
		rle_ids.extend([ids[i]] * len(mask_rle))

		if len(mask_rle) == 0:
			rles.extend([[1, 1]])
			rle_ids.extend([ids[i]])


test_ids = u.test_ids()
batch_size = 64
i = 0

id_batch = test_ids[i*batch_size : (i+1)*batch_size]

while i*batch_size < len(test_ids)-1:
	print(i)

	batch = u.test_imgs(for_ids=id_batch)
	ids = [i[:i.index('.')] for i in id_batch]

	# propagate the test images forward through the U-Net
	pred = model.predict(batch, batch_size=batch_size)[:,:,:,0]
	encode_batch(pred, ids)

	i += 1
	id_batch = test_ids[i*batch_size : (i+1)*batch_size]

df = pd.DataFrame()
df['id'] = rle_ids
df['rle_mask'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
df.to_csv(join('models', name, args.sub_name+'.csv'), index=False)
