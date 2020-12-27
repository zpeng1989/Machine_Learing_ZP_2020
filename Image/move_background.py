import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

rectangle = (0,56, 256, 150)

mask = np.zeros(image_rgb.shape[:2], np.uint8)

bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

cv2.grabCut(image_rgb, mask, rectangle, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask_2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

image_rgb_no = image_rgb*mask_2[:,:,np.newaxis]

plt.imshow(image_rgb_no)
plt.axis('off')
plt.show()

