import util as u
import os
from os.path import join

# Take the last 1/4th of the training images
# and move them to a directory called test-labeled
# so that we have a labeled test set to evaluate on

img_ids = u.img_ids(True)
m = int(0.75 * len(img_ids))
test_ids = img_ids[m:]

for i in test_ids:
	os.rename(join('data/train/images', i), join('data/test-labeled/images', i))
	os.rename(join('data/train/masks', i), join('data/test-labeled/masks', i))
