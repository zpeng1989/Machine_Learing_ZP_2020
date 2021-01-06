from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif

iris = load_iris()
features = iris.data
target = iris.target

features = features.astype(int)

chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)
print(features.shape[1])
print(features_kbest.shape[1])



fvalue_selector = SelectKBest(f_classif, k = 2)
features_kbest = fvalue_selector.fit_transform(features, target)

print(features.shape[1])
print(features_kbest.shape[1])




fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

print(features.shape[1])
print(features_kbest.shape[1])

