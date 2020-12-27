import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([50, 50, 50])
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
image_bgr_masked = cv2.bitwise_and(image, image, mask = mask)

image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)

#plt.imshow(image_rgb)
#plt.axis('off')
#plt.show()


image_grey = cv2.imread('plane.jpeg', cv2.IMREAD_GRAYSCALE)
max_out = 255
neighborhood_size = 99
subtract_mean = 5

image_bin = cv2.adaptiveThreshold(image_grey, max_out,
                                  #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  neighborhood_size,
                                  subtract_mean)
plt.imshow(image_bin)
plt.axis('off')
plt.show()


