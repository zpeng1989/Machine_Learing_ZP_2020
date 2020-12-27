import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_GRAYSCALE)

#plt.imshow(image, cmap='gray')
#plt.axis('off')
#plt.show()



print(type(image))
print(image)
print(image.shape)
print(image[0,0])

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)

print(image[0,0])



cv2.imwrite("panle_new.jpg", image)


## resize

image_50_50 = cv2.resize(image, (50, 50))
plt.imshow(image_50_50, cmap='gray')
plt.axis('off')
plt.show()


image_cut = image[:,:128]

plt.imshow(image_cut, cmap = 'gray')
plt.axis('off')
plt.show()



