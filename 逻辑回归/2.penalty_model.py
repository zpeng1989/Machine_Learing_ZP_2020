from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


iris = datasets.load_iris()
features = iris.data
target = iris.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
logistic_regression = LogisticRegressionCV(random_state=0, multi_class='ovr', penalty='l2', Cs=10)
model = logistic_regression.fit(features_standardized, target)

new_observation = [[.5, .5, .5, .5]]
print(model.predict(new_observation))
print(model.predict_proba(new_observation))


logistic_regression = LogisticRegression(random_state=0, solver='sag',class_weight='balanced')
model = logistic_regression.fit(features_standardized, target)

new_observation = [[.5, .5, .5, .5]]
print(model.predict(new_observation))
print(model.predict_proba(new_observation))

