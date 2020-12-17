import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures

feature = np.array([[43], [52432], [3], [1.2], [4324], [534]])
feature_a = np.array([[100], [23]])

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
scale_feature_model = minmax_scale.fit(feature)
scale_feature = scale_feature_model.transform(feature)
print(scale_feature_model.transform(feature_a))
print(scale_feature)

scaler = preprocessing.StandardScaler()
standardized_model = scaler.fit(feature)
print('SSSSSSSSSSSSSSSSSSS')
print(standardized_model.transform(feature))
print(standardized_model.transform(feature_a))

#############################

print('SSSSSSSSSSSSSSSSSSSSSSSSS')
features = np.array([[0.5, 0.3],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])
normalizer = Normalizer(norm ='l1')
print(normalizer.transform(features))

polynomial_interaction = PolynomialFeatures(degree= 2, include_bias=True)
polynomial_interaction_model = polynomial_interaction.fit(features)
print(polynomial_interaction_model.transform(features))





