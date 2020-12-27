import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('plane.jpeg', cv2.IMREAD_COLOR)

image_10 = cv2.resize(image,(10,10))
print(image_10.flatten())

#plt.imshow(image_10, cmap='gray')
#plt.axis('off')
#plt.show()

channels = cv2.mean(image_10)
observation = np.array([(channels[2], channels[1], channels[0])])
print(observation)


features = []



colors = ['r', 'g', 'b']
for i, channel in enumerate(colors):
    histogram = cv2.calcHist([image], [i], None, [256], [0,256])
    features.extend(histogram)

observation = np.array(features).flatten()

print(observation)



