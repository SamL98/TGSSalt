import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os.path import join
import sys

_stdout = sys.stdout
_stderr = sys.stderr

def silence():
	sys.stdout = open('/dev/null', 'w')
	sys.stderr = open('/dev/null', 'w')

def unsilence():
	sys.stdout = _stdout
	sys.stderr = _stderr

from scipy.misc import imread
from skimage.transform import resize
from cv2 import fastNlMeansDenoisingColored
import numpy as np

_data_dir = 'data'
_train_dir = join(_data_dir, 'train')
_test_lab_dir = join(_data_dir, 'test-labeled')
_test_unlab_dir = join(_data_dir, 'test-unlabeled')

_img_folder = 'images'
_mask_folder = 'masks'

_train_ids = []
_test_lab_ids = []
_test_unlab_ids = []

_gray = True
_cs = False
_dim = 128
_den = False
_ret_depth = False
_attrs = {
	'gray': _gray,
	'cs': _cs,
	'dim': _dim,
	'den': _den
}

def set_gray(gray):
	global _gray
	_gray = gray

def set_cs(cs):
	global _cs

	_cs = cs
	if _cs: set_gray(False)

def set_dim(dim):
	global _dim
	_dim = dim

def set_den(den):
	global _den
	_den = den

def set_depth(dep):
	global _ret_depth
	_ret_depth = dep

def set_attrs(attrs):
	global _attrs
	for k, v in attrs.items():
		_attrs[k] = v

"""
Return the filenames for the given dataset
:param train: whether or not to use the training dataset
"""
def img_ids(train=True):
	if train:
		global _train_ids
		if len(_train_ids) > 0: return _train_ids
		_train_ids = os.listdir(join(_train_dir, _img_folder))
		return _train_ids
	else:
		global _test_lab_ids
		if len(_test_lab_ids) > 0: return _test_lab_ids
		_test_lab_ids = os.listdir(join(_test_lab_dir, _img_folder))
		return _test_lab_ids

"""
Return the filenames for the test dataset (the ones to create a submission for)
"""
def test_ids():
	global _test_unlab_ids
	if len(_test_unlab_ids) > 0: return _test_unlab_ids
	_test_unlab_ids = os.listdir(join(_test_unlab_dir, _img_folder))
	return _test_unlab_ids

"""
Return the correct image directory
:param train: whether or not to return the training dataset directory
:param mask: whether or not to return the mask directory
"""
def _get_dir(train, mask):
	init_dir, ext_dir = _train_dir, _img_folder
	if not train: init_dir = _test_lab_dir
	if mask: ext_dir = _mask_folder
	return join(init_dir, ext_dir)

"""
Read an image and resize it to 128x128 (size of U-Net).
Divide by the maximum so that image is in range [0, 1].
Don't divide if the image is all 0's (black).

:param f: filename of the image
"""
import pandas as pd
_depths = None

def load_depths():
	global _depths
	_depths = pd.read_csv('data/depths.csv', index_col='id')
	return _depths

def _read_img(f):
	global _gray, _cs, _dim, _den

	# read the image and resize it to a size the U-Net can deal with
	img = resize(imread(f), (_dim, _dim), cval=0, mode='constant', preserve_range=True)
	
	# return if the image 2d (i.e. it is a mask)
	if len(img.shape) == 2:
		if img.max() > 0.0: return img / float(img.max())
		return img

	if _den: img = fastNlMeansDenoisingColored(img.astype(np.uint8), 10, 10, 7, 21).astype(np.float32)

	# if we want either gray or cumsum-concat images,
	# get the first color channel (all three are them same)
	if (_gray or _cs): img = img[:,:,0]

	# if the image is not blank, divide by it's maximum (either 255 or 65535)
	if img.max() > 0.0: img /= float(img.max())

	# at this point, return if we don't want cumsum-concat
	if not _cs: return img

	b = (128-101)//2 # border of the image after resize to 128x128

	# not sure how this helps but people on Kaggle are doing it
	img_csum = (img - img[b:-b, b:-b].mean()).cumsum(axis=0)
	img_csum -= img_csum[b:-b, b:-b].mean()
	img_csum /= max(1e-3, img_csum[b:-b, b:-b].std())

	if img_csum.max() > 0.0 and img_csum.min() < img_csum.max():
		img_csum = (img_csum - img_csum.min()) / (img_csum.max() - img_csum.min())

	# concatenate the cumsum as a second channel
	img = img[:,:,np.newaxis]
	img_csum = img_csum[:,:,np.newaxis]

	return np.concatenate((img, img_csum), axis=2)

def _get_depth(img_id):
	global _depths
	if _depths is None: load_depths()

	img_id = img_id[:img_id.index('.')]
	return _depths.loc[img_id, 'z']

"""
Return an array of all images or masks in the given dataset
:param train: whether or not to return images from the training dataset
:param mask: whether or not to return the masks from the given dataset
"""
def _imgs_or_masks(mask, for_ids=None):
	if for_ids is None: ids = img_ids(True)
	else: ids = for_ids

	d = _get_dir(True, mask) # get the correct directory to look in
	arr = np.array([_read_img(join(d, i)) for i in ids])

	global _ret_depth
	if _ret_depth and not mask:
		return arr, np.array([_get_depth(i) for i in ids])

	return arr

"""
Return an image or mask for the given id
:param img_id: the id of the image to return
:param train: which dataset to read from
:param mask: whether or not read a mask
"""
def _img_or_mask(img_id, train, mask):
	d = _get_dir(train, mask)
	return _read_img(join(d, img_id))

"""
Return an image for the given id
:param img_id: id of the image to load
:param train: which dataset it's in
"""
def img_by_id(img_id, train=True):
	return _img_or_mask(img_id, train, False)

"""
Return a mask for the given id
:param img_id: the id of the mask to load
:param train: which dataset to use
"""
def mask_by_id(img_id, train=True):
	return _img_or_mask(img_id, train, True)

"""
Return an array of all the unlabeled test images
"""
def test_imgs(for_ids=None):
	if for_ids is None: ids = test_ids()
	else: ids = for_ids

	d = join(_test_unlab_dir, _img_folder) # get the correct directory
	return np.array([_read_img(join(d, i)) for i in ids])

"""
Return the training or testing images
:param train: whether or not to return images from the training dataset
:param gray: whether or not to return grayscale
"""
def imgs(for_ids=None):
	return _imgs_or_masks(False, for_ids)

"""
Return the training or testing masks
:param train: whether or not to return masks from the training dataset
:param gray: whether or not to return grayscale
"""
def masks(for_ids=None):
	return _imgs_or_masks(True, for_ids)

"""
Return the images and masks for the given dataset
:param train: whether or not to return images and masks from the training dataset
"""
def ips(for_ids=None):
	return imgs(for_ids), masks(for_ids).astype(np.uint8)

"""
Return a random unlabeled test image
"""
def rand_test_img():
	ids = test_ids()
	img_id = np.random.choice(ids)
	d = join(_test_unlab_dir, _img_folder)
	return _read_img(join(d, img_id))

"""
Return a random image or mask, depending on parameters
:param train: which dataset to return from
:param mask: whether or not to return a random mask
"""
def _rand_img_or_mask(train, mask):
	ids = img_ids(train=train) # get the filenames for the dataset
	img_id = np.random.choice(ids) # select a random id
	d = _get_dir(train, mask) # get the correct directory
	return _read_img(join(d, img_id))

"""
Return a random image
:param train: which dataset to return from
"""
def rand_img(train=True):
	return _rand_img_or_mask(True, False)

"""
Return a random mask
:param train: which dataset to return from
"""
def rand_mask(train=True):
	return _rand_img_or_mask(train, True)	

"""
Return a random image, mask pair
:param train: which dataset to return from
"""
def rand_ip(train=True):
	ids = img_ids(train=train)
	img_id = np.random.choice(ids)
	img = _read_img(join(_get_dir(train, False), img_id))
	mask = _read_img(join(_get_dir(train, True), img_id))
	return img, mask

def disp_img(img):
	import matplotlib.pyplot as plt
	plt.imshow(img, 'gray')
	plt.show()

"""
Display, using matplotlib, a given image, mask pair
:param img: the image to display
:param mask: the mask to display 
"""
def disp_ip(img, mask, wbutton=False, update_func=None):
	import matplotlib.pyplot as plt
	from disp_util import add_button_to

	_, a = plt.subplots(1, 2)

	a[0].imshow(img, 'gray')
	a[0].set_title('Image')

	a[1].imshow(mask, 'gray')
	a[1].set_title('Mask')

	#print(update_func)
	def update():
		print('update')
		img, mask = fn()
		a[0].set_data(img)
		a[1].set_data(img)


	if wbutton:
		#print('weroiu')
		add_button_to(plt, update)

	plt.show()	

"""
Display an image, mask, and predicted mask triplet
:param img: original image
:param mask: ground truth mask
:param pred: predicted mask
"""
def disp_imp(img, mask, pred, titles=None):
	if titles is None:
		titles = ['Image', 'Mask', 'Pred']

	import matplotlib.pyplot as plt
	_, a = plt.subplots(1, 3)

	a[0].imshow(img, 'gray')
	a[0].set_title(titles[0])

	a[1].imshow(mask, 'gray')
	a[1].set_title(titles[1])

	a[2].imshow(pred, 'gray')
	a[2].set_title(titles[2])

	plt.show()	

def disp_2x2(imgs, titles=None):
	import matplotlib.pyplot as plt
	_, a = plt.subplots(2, 2)

	a[0,0].imshow(imgs[0], 'gray')
	a[0,1].imshow(imgs[1], 'gray')
	a[1,0].imshow(imgs[2], 'gray')
	a[1,1].imshow(imgs[3], 'gray')

	a[0,0].axis('off')
	a[0,1].axis('off')
	a[1,0].axis('off')
	a[1,1].axis('off')

	if not titles is None:
		a[0,0].set_title(titles[0])
		a[0,1].set_title(titles[1])
		a[1,0].set_title(titles[2])
		a[1,1].set_title(titles[3])

	plt.show()	

from keras.models import model_from_json
def load_unet(name='unet'):
	with open(join('models', name, 'model.json')) as f:
		json = f.read()
	model = model_from_json(json)
	model.load_weights(join('models', name, 'model.h5'))
	return model

from skimage.morphology import label
import cv2 as cv

def postprocessing(mask):
	comps = label(mask)

	for i in range(1, len(np.unique(comps))):
		if np.sum(comps==i) < 300:
			mask[comps==i] = 0

	for i in range(5):
		mask = cv.medianBlur(mask, 5)

	return mask

from PIL import ImageEnhance, Image

def preprocess(img, sharpen=False, contrast=True):
	if not img.dtype == np.uint8:
		img = (img*255).astype(np.uint8)
	img = Image.fromarray(img)

	if contrast:
		contrast = ImageEnhance.Contrast(img)
		img = contrast.enhance(2.0)

	if sharpen:
		sharp = ImageEnhance.Sharpness(img)
		img = sharp.enhance(2.0)

	return np.array(img)/255.