from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

iris = datasets.load_iris()
features = iris.data
target = iris.target

classifer = GaussianNB()
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method = 'sigmoid')

model_clf = classifer_sigmoid.fit(features, target)

new_observation = [[2.6,2.6,2.6,0.4]]

print(classifer_sigmoid.predict_proba(new_observation))
print(model_clf.predict_proba(new_observation))
