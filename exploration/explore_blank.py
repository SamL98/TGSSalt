import util as u
import numpy as np
import matplotlib.pyplot as plt

with open('blanks_te.txt') as f:
	ids = f.read().split('\n')

ids = np.random.choice(ids, 5, replace=False)
for id in ids:
	img = u.img_by_id(id, train=False)
	plt.imshow(img, 'gray')
	plt.show()
