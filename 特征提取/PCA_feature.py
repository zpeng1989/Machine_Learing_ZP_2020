from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
digits_str = digits.data
features_model = StandardScaler().fit(digits_str)
features = features_model.transform(digits_str)

#features = StandardScaler.fit_transform(digits_str)

pca = PCA(n_components= 0.99, whiten = True)
features_pca = pca.fit_transform(features)

print(features.shape[1])
print(features_pca.shape[1])


