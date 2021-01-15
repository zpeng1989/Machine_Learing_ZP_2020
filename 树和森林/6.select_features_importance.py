import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)

model = randomforest.fit(features, target)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)
print(importances)

names = [iris.feature_names[i] for i in indices]

plt.figure()

plt.title('features importance')

plt.bar(range(features.shape[1]), importances[indices])
plt.xticks(range(features.shape[1]), names, rotation= 90)

plt.show()







