from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()

features_standardized = scaler.fit_transform(features)
svc = SVC(kernel = 'linear', probability = True, random_state=0)

model = svc.fit(features_standardized, target)
new_observation= [[.4, .4, .4, .4]]
print(model.predict_proba(new_observation))




