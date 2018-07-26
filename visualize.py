import util
from scipy.signal import convolve2d
from scipy.special import expit
import numpy as np

model = util.load_unet()
first_layer = model.layers[1]
theta = first_layer.get_weights()
Ws, bs = theta[0], theta[1]

util.set_cs(False)

img = util.rand_img()

def disp_filter(filt_num):
    global Ws, bs, img
    W, b = Ws[:,:,0,filt_num], bs[filt_num]

    res = convolve2d(img, W, mode='same')+b
    res = np.maximum(res, 0)
    #res = expit(res)

    util.disp_ip(img, res)

for i in range(Ws.shape[-1]):
    disp_filter(i)