from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
features = iris.data
target = iris.target

classifer = GaussianNB()
model = classifer.fit(features, target)

new_observation = [[4,5,5,0.4]]
print(model.predict(new_observation))

clf = GaussianNB(priors= [0.25, 0.25, 0.5])

model = clf.fit(features, target)

print(model.predict(new_observation))

