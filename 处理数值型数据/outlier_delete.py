import numpy as np

from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

features, _ = make_blobs(n_samples=10, n_features = 2, centers = 1, random_state = 1)

print(features)

features[2,0] = 10000
features[2,1] = 10000

outlizer_detector = EllipticEnvelope(contamination=.1)
print(outlizer_detector.fit(features))

print(outlizer_detector.predict(features))


feature = features[:,0]

def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 -(iqr*1.5)
    high_bound = q3 + (iqr*1.5)
    return np.where((x>high_bound)|(x < lower_bound))

print(indicies_of_outliers(feature))

