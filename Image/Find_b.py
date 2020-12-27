import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)

median_intensity = np.median(image)

low_threshold = int(max(0,(1-0.33)*median_intensity))
up_threshold = int(min(155,(1+0.33)*median_intensity))

image_canny = cv2.Canny(image, low_threshold, up_threshold)

#plt.imshow(image_canny, cmap='gray')
#plt.axis('off')
#plt.show()

# jiao

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = np.float32(image_gray)

block_size = 2
aperture = 29
free_paramter = 0.04

detector_response = cv2.cornerHarris(image_gray, block_size, aperture, free_paramter)

detector_response = cv2.dilate(detector_response, None)

threshold = 0.02
image[detector_response > threshold*detector_response.max()] = [255, 255, 255]

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(detector_response, cmap='gray')
plt.axis('off')
plt.show()



