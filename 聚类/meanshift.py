from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift, DBSCAN

iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = MeanShift()
cluster_model = cluster.fit(features_std)

print(cluster_model.labels_)

cluster = DBSCAN()

model = cluster.fit(features_std)


print(model.labels_)
