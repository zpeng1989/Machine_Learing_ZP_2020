import numpy as np
import pandas as pd

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



######

houses = pd.DataFrame()
houses['Price'] = [43245,53242,13234,432455]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

print(houses[houses['Bathrooms']<20])

houses["Outlizer"] = np.where(houses['Bathrooms']<20, 0, 1)
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

print(houses)




