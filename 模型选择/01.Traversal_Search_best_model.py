import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
features = iris.data
target = iris.target


logistic = linear_model.LogisticRegression(max_iter=1000)
penalty = ['l1', 'l2']
C = np.logspace(0,4,10)
print(C)
hyperparamenters = dict(C = C, penalty = penalty)
print(hyperparamenters)
gridsearch = GridSearchCV(logistic, hyperparamenters, cv=5, verbose=0)

best_model = gridsearch.fit(features, target)
print(best_model.best_estimator_.get_params())
print('sssssssss')
print(best_model.predict(features))