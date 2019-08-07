import cv2
import numpy as np
from matplotlib import pyplot as plt

# simple averaging filter without scaling parameter
#mean_filter = np.ones((3,3))

# creating a guassian filter
x = cv2.getGaussianKernel(5,2.5)
gaussian = x*x.T

# different edge detecting filters
# scharr in x-direction
dog = np.array([[-0.0625, 0., -0.0625],
[0.,0.25,0.],
[-0.0625, 0., -0.0625]])
# sobel in x direction
sobel_x= np.array([[-1., 0., 1.],
[-2., 0., 2.],
[-1., 0., 1.]])
# sobel in y direction
sobel_y= np.array([[-1.,-2.,-1.],
[0, 0., 0],
[1., 2., 1.]])
# laplacian
laplacian=np.array([[0, 1., 0],
[1.,-4., 1.],
[0, 1., 0.]])
# LoG
lapoG=np.array([[-1, 2., -1],
[2,-4., 2.],
[-1, 2., -1]])

filters = [gaussian, laplacian, sobel_x, sobel_y, dog, lapoG]
filter_name = ['gaussian','laplacian', 'sobel_x',
'sobel_y', 'Dog', 'LoG']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z)+1) for z in fft_shift]

for i in range(6):
	
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i], interpolation='bicubic', cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])

plt.show()