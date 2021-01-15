import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel

iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)

selector = SelectFromModel(randomforest, threshold=0.3)

features_important_model = selector.fit(features, target)
features_important = features_important_model.transform(features)

model = randomforest.fit(features_important, target)

observation = [[5,4,5,3]]

print(model.predict(features_important_model.transform(observation)))


