from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = datasets.load_iris()
features = iris.data

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

cluster = KMeans(n_clusters=3, random_state=0)
model = cluster.fit(features_std)

print(model.labels_)
print(iris.target)

new_observation= [[0.8, 0.8, 0.8, 0.8]]

print(model.predict(new_observation))

print(model.cluster_centers_)
