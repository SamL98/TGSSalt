import util as u
import numpy as np

u.set_gray(True)

for _ in range(5):
    img = (u.rand_img()*255).astype(np.uint8)
    sharpened = u.preprocess(img, sharpen=False, contrast=True)
    fully_processed = u.preprocess(img, sharpen=True, contrast=True)
    u.disp_imp(img/255., sharpened/255., fully_processed/255.)