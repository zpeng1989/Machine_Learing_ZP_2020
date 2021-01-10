from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

boston = load_boston()
features = boston.data
target = boston.target

scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

regression = Ridge(alpha=0.5)
model = regression.fit(features_standardized, target)

print(model.coef_)

regr_cv = RidgeCV(alphas = [0.1, 1, 10])

model_cv = regr_cv.fit(features_standardized, target)

print(model_cv.coef_)
print(model_cv.alpha_)

regression = Lasso(alpha=0.5)
model = regression.fit(features_standardized, target)

print(model.coef_)



