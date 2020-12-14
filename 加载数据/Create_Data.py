from sklearn.datasets import make_regression

features, target, coefficients = make_regression(n_samples=100, n_features = 3, n_informative= 3,n_targets=1, noise =0, coef = True, random_state = 1)
print(features[:3])
print(target[:3])

from sklearn.datasets import make_classification
features, target = make_classification(n_samples=100, n_features=3, n_informative=3, n_redundant=0,n_classes=2, weights=[.25, .75], random_state=1)

print(features[:3])
print(target[:3])

from sklearn.datasets import make_blobs
features, target = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=1)
print(features[:3])
print(features[:3,1])

