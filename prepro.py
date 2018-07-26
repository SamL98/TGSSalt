import util as u
import numpy as np
from scipy.signal import convolve2d

kernel_y = np.array([
    [3., 10., 3.],
    [0., 0., 0.],
    [-3., -10., -3.]
])
kernel_x = kernel_y.T

for _ in range(10):
    img, mask = u.rand_ip()
    edges_y = convolve2d(img, kernel_y, mode='same')
    edges_x = convolve2d(img, kernel_x, mode='same')
    u.disp_imp(img*255, edges_x+edges_y, mask)