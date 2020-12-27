import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
image_blurry = cv2.blur(image, (5,5))
plt.imshow(image_blurry, cmap='gray')
plt.axis('off')
plt.show()

## kernel

kernel = np.ones((5,5))/25

print(kernel)

image_kernal = cv2.filter2D(image, -1, kernel)
plt.imshow(image_kernal, cmap = 'gray')
plt.xticks([])
plt.yticks([])
plt.show()


