from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = datasets.load_iris()

features = iris.data#[:100,:]
target = iris.target#[:100]

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
svc = SVC(kernel = 'linear', random_state = 0)
model = svc.fit(features_standardized, target)

print(model.support_vectors_)
print(model.support_)
print(model.n_support_)

