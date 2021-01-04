from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np

digits = datasets.load_digits()

features = StandardScaler().fit_transform(digits.data)
#print(features.shape)
#print(features[1:3,])
features_sparse = csr_matrix(features)
#print(features_sparse)

tsvd = TruncatedSVD(n_components = 10)

features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)

print(features_sparse.shape[1])
print(features_sparse_tsvd.shape[1])

#####

print(tsvd.explained_variance_ratio_[1:8].sum())
