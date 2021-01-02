from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles

features, _ = make_circles(n_samples = 1000, random_state = 1, noise = 0.1)

kpca = KernelPCA(kernel= 'rbf', gamma=15, n_components=1)
features_kpca_model = kpca.fit(features)
features_kpca = features_kpca_model.transform(features)

print(features.shape[1])
print(features_kpca.shape[1])


####

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()
features = iris.data
target = iris.target

lda = LinearDiscriminantAnalysis(n_components = 1)
feature_lda = lda.fit(features, target).transform(features)

print(features.shape[1])
print(feature_lda.shape[1])
lda_var_ratios = lda.explained_variance_ratio_

def select_n_components(var_ratio, goal_var):
    total_variance = 0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break

    return n_components

print(select_n_components(lda_var_ratios, 0.95))

from sklearn.decomposition import NMF
from sklearn import datasets

digits = datasets.load_digits()
features = digits.data

nmf = NMF(n_components=10, random_state = 1)
features_nmf = nmf.fit_transform(features)

print(features.shape[1])
print(features_nmf.shape[1])






