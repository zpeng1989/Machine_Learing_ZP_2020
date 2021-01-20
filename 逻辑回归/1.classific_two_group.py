from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
features = iris.data[:100, :]
target = iris.target[:100]

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
logistic_regression = LogisticRegression(random_state=0)
model = logistic_regression.fit(features_standardized, target)

new_observation = [[.5, .5, .5, .5]]
print(model.predict(new_observation))

iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
logistic_regression = LogisticRegression(random_state=0, multi_class='ovr')
model = logistic_regression.fit(features_standardized, target)

new_observation = [[.5, .5, .5, .5]]
print(model.predict(new_observation))
print(model.predict_proba(new_observation))

