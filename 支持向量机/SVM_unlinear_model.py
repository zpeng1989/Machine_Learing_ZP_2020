from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(0)

features = np.random.randn(200, 2)
target_xor = np.logical_xor(features[:,0]>0, features[:,1]>0)
target = np.where(target_xor, 0, 1)

svc = SVC(kernel = 'rbf', random_state=0, gamma=1, C=1)

model = svc.fit(features, target)

print(model.predict(features))



