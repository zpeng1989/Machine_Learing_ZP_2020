import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(1)

iris = datasets.load_iris()
features = iris.data
target = iris.target

preprocess = FeatureUnion([("std", StandardScaler()),("pca", PCA())])

pipe = Pipeline([("preprocess", preprocess),("classifier", LogisticRegression(max_iter=1000))])


search_space = [{"preprocess__pca__n_components":[1,2,3],
                # "classifier__penalty":[None, 'l2'],
                 "classifier__C":np.logspace(0,4,10)}]

clf = GridSearchCV(pipe, search_space, cv = 5, verbose = 0, n_jobs = -1)

best_model = clf.fit(features, target)
print(best_model.best_estimator_.get_params())

