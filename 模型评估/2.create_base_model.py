from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

boston = load_boston()
#print(boston)
#print(type(boston))
features, target = boston.data, boston.target
features_train, features_test, target_train, target_test = train_test_split(features, target)

dummy = DummyRegressor(strategy='mean')
dummy.fit(features_train, target_train)
print(dummy.score(features_test, target_test))

from sklearn.linear_model import LinearRegression

ols = LinearRegression()
ols.fit(features_train, target_train)
print(ols.score(features_test, target_test))



