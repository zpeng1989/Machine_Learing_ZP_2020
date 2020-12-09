import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed()

iris = datasets.load_iris()
features = iris.data
target = iris.target

pipe = Pipeline([('classifier', RandomForestClassifier())])

search_space = [{'classifier':[LogisticRegression(max_iter=1000)], 'classifier__penalty':['l1', 'l2'], 'classifier__C':np.logspace(0,4,10)},
                {'classifier':[RandomForestClassifier()], 'classifier__n_estimators':[10,100,1000], 'classifier__max_features':[1,2,3]}]


gridsearch = GridSearchCV(pipe, search_space, cv = 5, verbose= 0, n_jobs = -1)
best_model = gridsearch.fit(features, target)

print(best_model.best_estimator_.get_params())