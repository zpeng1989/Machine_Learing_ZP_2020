import cv2
import numpy as np

from matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
#image = cv2.imread('plane.jpeg', cv2.IMREAD_GRAYSCALE)
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
image_shape = cv2.filter2D(image, -1, kernel)

image = cv2.imread('plane.jpeg', cv2.IMREAD_GRAYSCALE)
image_enhanced = cv2.equalizeHist(image)

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
image_yuv[:,:,0] = cv2.equalizeHist(image_yuv[:,:,0])
image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)




plt.imshow(image_rgb, cmap='gray')
plt.axis('off')
plt.show()



