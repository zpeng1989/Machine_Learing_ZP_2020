from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
festures = iris.data

standardizer = StandardScaler()

features_standardizer = standardizer.fit_transform(festures)
nearest_neighbors = NearestNeighbors(n_neighbors = 2).fit(features_standardizer)
new_observation = [1.,1.,1.,1.]

distance, indices = nearest_neighbors.kneighbors([new_observation])

print(features_standardizer[indices])
print(distance)

nearest_neighbors_euclidean = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features_standardizer)
nearest_neighbors_with_self = nearest_neighbors_euclidean.kneighbors_graph(features_standardizer).toarry()

for i,x in enumerate(nearest_neighbors_with_self):
    x[i] = 0

print(nearest_neighbors_with_self[0])
