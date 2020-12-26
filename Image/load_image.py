import cv2
import numpy as np

form matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()