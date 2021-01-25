from sklearn import preprocessing
import numpy as np

features = np.array([[-100,3240.1], [-200.2, -234.1], [5000.5, 150.1], [6000.6, -125.1]])

scaler = preprocessing.StandardScaler()
features_standoardized = scaler.fit_transform(features)

print(features_standoardized)

print(round(features_standoardized[:,0].mean()))
print(features_standoardized[:,0].std())
