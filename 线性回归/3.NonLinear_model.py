from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
features = boston.data[:,0:1]
target = boston.target

polynomial = PolynomialFeatures(degree = 3, include_bias = False)
features_polynomial = polynomial.fit_transform(features)

regression = LinearRegression()
model = regression.fit(features_polynomial, target)

print(features[0])
print(features[0]**2)
print(features[0]**3)

print(features_polynomial[0])
