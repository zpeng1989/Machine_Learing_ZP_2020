from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

iris = datasets.load_iris()
features = iris.data
target = iris.target

randomforest = RandomForestClassifier(random_state = 0, n_jobs = -1, criterion = 'entropy')

model = randomforest.fit(features, target)

observation = [[5,4,5,3]]
print(model.predict(observation))
