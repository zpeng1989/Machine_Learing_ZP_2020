import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model


warnings.filterwarnings(action = 'ignore', module = 'scipy', message = '^internal gelsd')

features, target = make_regression(n_samples = 10000, n_features = 100, n_informative = 2, random_state = 1)

ols = linear_model.LinearRegression()

rfecv = RFECV(estimator=ols, step=1, scoring = 'neg_mean_squared_error')

print(rfecv.fit(features, target))
print(rfecv.transform(features))
