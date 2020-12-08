from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

features, target = make_regression(n_samples = 10000,
                                      n_features = 3,
                                      n_informative=3,
                                      n_targets = 1, 
                                      noise = 50,
                                      coef = False,
                                      random_state = 1)

ols = LinearRegression()
print(cross_val_score(ols, features, target, scoring='neg_mean_squared_error'))


print(cross_val_score(ols, features, target, scoring = 'r2'))