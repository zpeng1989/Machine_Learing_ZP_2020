from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state = 0,criterion='entropy')

model = decisiontree.fit(features, target)

observation = [[5,4,3,2]]

print(model.predict(observation))
print(model.predict_proba(observation))

