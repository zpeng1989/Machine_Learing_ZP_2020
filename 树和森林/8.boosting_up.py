from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

adaboost = AdaBoostClassifier(random_state = 0)
model = adaboost.fit(features, target)