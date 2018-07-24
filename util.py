import os
from os.path import join

from scipy.misc import imread
from skimage.transform import resize
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

def set_gray(gray):
	global _gray
	_gray = gray

def set_cs(cs):
	global _cs

	_cs = cs
	if _cs: set_gray(False)

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
def _read_img(f):
	global _gray, _cs

	# read the image and resize it to a size the U-Net can deal with
	img = resize(imread(f), (128, 128), mode='constant', preserve_range=True)

	# return if the image 2d (i.e. it is a mask)
	if len(img.shape) == 2: return img

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

	# concatenate the cumsum as a second channel
	img = img[:,:,np.newaxis]
	img_csum = img_csum[:,:,np.newaxis]
	return np.concatenate((img, img_csum), axis=2)

"""
Return an array of all images or masks in the given dataset
:param train: whether or not to return images from the training dataset
:param mask: whether or not to return the masks from the given dataset
"""
def _imgs_or_masks(train, mask):
	ids = img_ids(train) # get the image filenames
	d = _get_dir(train, mask) # get the correct directory to look in
	return np.array([_read_img(join(d, i)) for i in ids])

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
def test_imgs():
	ids = test_ids() # get the image filenames
	d = join(_test_unlab_dir, _img_folder) # get the correct directory
	return np.array([_read_img(join(d, i)) for i in ids])

"""
Return the training or testing images
:param train: whether or not to return images from the training dataset
:param gray: whether or not to return grayscale
"""
def imgs(train=True):
	return _imgs_or_masks(train, False)

"""
Return the training or testing masks
:param train: whether or not to return masks from the training dataset
:param gray: whether or not to return grayscale
"""
def masks(train=True):
	return _imgs_or_masks(train, True)

"""
Return the images and masks for the given dataset
:param train: whether or not to return images and masks from the training dataset
"""
def ips(train=True):
	return imgs(train), masks(train).astype(np.uint8)

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
	return _rand_img_or_mask(train, False)

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

"""
Display, using matplotlib, a given image, mask pair
:param img: the image to display
:param mask: the mask to display 
"""
def disp_ip(img, mask):
	import matplotlib.pyplot as plt
	_, a = plt.subplots(1, 2)

	a[0].imshow(img, 'gray')
	a[0].set_title('Image')

	a[1].imshow(mask, 'gray')
	a[1].set_title('Mask')

	plt.show()	

"""
Display an image, mask, and predicted mask triplet
:param img: original image
:param mask: ground truth mask
:param pred: predicted mask
"""
def disp_imp(img, mask, pred):
	import matplotlib.pyplot as plt
	_, a = plt.subplots(1, 3)

	a[0].imshow(img, 'gray')
	a[0].set_title('Image')

	a[1].imshow(mask, 'gray')
	a[1].set_title('Mask')

	a[2].imshow(pred, 'gray')
	a[2].set_title('Pred')

	plt.show()	
