from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()

features, target = iris.data, iris.target

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)

dummy = DummyClassifier(strategy = 'uniform', random_state = 1)
dummy.fit(features_train, target_train)
print(dummy.score(features_test, target_test))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(features_train, target_train)
print(classifier.score(features_test, target_test))



