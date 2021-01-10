from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

regression = LinearRegression()

model = regression.fit(features, target)

print(model)
print(model.intercept_)
print(model.coef_)
