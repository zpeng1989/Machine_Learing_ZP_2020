import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

feature = np.array([['Texas'], ['California'], ['Texas'], ['Texas'], ['Delawre']])
one_hot = LabelBinarizer()
print(one_hot.fit_transform(feature))
print(one_hot.classes_)

import pandas as pd

print(pd.get_dummies(feature[:,0]))



multiclass_feature = [('Texas', 'Florida'), ('California', 'Alabama'), ('Texas', 'Florida'), ('Delware', 'Florida'), ('Texas', 'Alabama')]
one_hot_multiclass = MultiLabelBinarizer()
print(one_hot_multiclass.fit_transform(multiclass_feature))

#### ordinal class

import pandas as pd

dataframe = pd.DataFrame({'Score':["Low", "Low", "Median", "Median", "High"]})

sacle_mapper = {'Low':1, 'Median':2, 'High':3}

print(dataframe['Score'].replace(sacle_mapper))


#### dict

from sklearn.feature_extraction import DictVectorizer

data_dict = [{'Red':2, 'Blue':4},
             {'Red':4, 'Blue':3},
             {'Red':1, 'Yellow':2},
             {'Red':2, 'Yellow':2}]

dictvectorizer = DictVectorizer(sparse = False)
feastures_model = dictvectorizer.fit(data_dict)
print(feastures_model.transform(data_dict))

feature_names = dictvectorizer.get_feature_names()
print(feature_names)

#### nan

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[0, 2.10, 1.45], [1, 1.18, 1.33], [0, 1.22, 1.27], [1, -0.21, -1.19]])

X_with_nan = np.array([[np.nan, 0.87, 1.32], [np.nan, -0.67, -0.22]])

clf = KNeighborsClassifier(3, weights = 'distance')
trained_model = clf.fit(X[:,1:],X[:,0])

imputed_values = trained_model.predict(X_with_nan[:,1:])

print(imputed_values)






