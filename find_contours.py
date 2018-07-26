from skimage.measure import find_contours
from skimage.filters import frangi, hessian

import cv2 as cv
import numpy as np
import util as u

model = u.load_unet()

bin_width = 0.15

for _ in range(5):
    img, mask = u.rand_ip()
    proc = u.preprocess(img)

    pred = model.predict(np.array([np.expand_dims(img, axis=2)]))[0,:,:,0]
    pred = (pred > 0.5).astype(np.uint8)
    pts = np.argwhere(pred > 0)

    if len(pts) > 0:
        t, l = pts[:,0].min(), pts[:,1].min()+1
        b, r = pts[:,0].max(), pts[:,1].max()+1

        padding = 25
        t, l = max(t-padding, 0), max(l-padding, 0)
        b, r = min(b+padding, img.shape[0]), min(r+padding, img.shape[1])
    else:
        t, l, b, r = 0, 0, 1, 1

    cropped = np.zeros_like(proc)
    cropped[t:b, l:r] = proc[t:b, l:r]

    #thresh = cv.threshold(cropped, 0.75, 1.0, cv.THRESH_BINARY)[1]

    '''
    thresh = np.zeros_like(proc)
    cc = cropped.copy()

    curr_bin = 1.0
    while curr_bin > bin_width:
        pts = np.argwhere(cc-bin_width > curr_bin)
        cc[cc > curr_bin+bin_width] = 0

        for pt in pts:
            thresh[pt[0], pt[1]] = curr_bin
        
        curr_bin -= bin_width'''
    
    thresh = frangi(cropped, beta1=20.0, beta2=5.0, black_ridges=False)

    u.disp_2x2([img, cropped, thresh, mask], titles=['Image', 'Image+Contrast', 'Thresholded', 'Mask'])