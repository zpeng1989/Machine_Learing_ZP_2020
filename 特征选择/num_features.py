from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

iris = datasets.load_iris()

features = iris.data
target = iris.target

thresholder = VarianceThreshold(threshold=0.5)

features_high_variance = thresholder.fit_transform(features)
print(features_high_variance[0:3])

print(thresholder.fit(features).variances_)

