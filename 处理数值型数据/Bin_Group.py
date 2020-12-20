import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import Binarizer
#from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
##from sklearn.preprocessing import Imputer

age = np.array([[6],[12],[20],[36],[65]])

binarizer = Binarizer(18)
print(binarizer.fit_transform(age))

print(np.digitize(age, bins=[20,30,64]))


#### cluster

features, _ = make_blobs(n_samples = 50, n_features = 2, centers = 3, random_state = 1)

dataframe = pd.DataFrame(features, columns = ['feature_1', 'feature_2'])
clusterer = KMeans(3, random_state = 0)
clu_model = clusterer.fit(features)
dataframe['groups'] = clu_model.predict(features)

print(dataframe.head(5))


### delte

features = np.array([[1.1, 11.1], [2.2, 22.2], [3.3, 33.3], [4.4, 44.4], [np.nan, 55]])

print(features[~np.isnan(features).any(axis=1)])
dataframe = pd.DataFrame(features, columns = ['featutre1', 'feature2'])

print(dataframe.dropna())



### imputer

features, _ = make_blobs(n_samples = 1000, n_features = 2, random_state = 1)

scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)


true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan


print(standardized_features)

##mean_imputer = Imputer(strategy = 'mean', axis = 0)

##features_mean_imputed = mean_imputer.fit_transform(standardized_features)

##print(features_mean_imputed[0,0])
