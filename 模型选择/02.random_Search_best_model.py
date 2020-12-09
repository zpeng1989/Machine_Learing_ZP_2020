from scipy.stats import uniform
from sklearn import datasets, linear_model
from sklearn.model_selection import RandomizedSearchCV

iris = datasets.load_iris()
features = iris.data
target = iris.target

logistic = linear_model.LogisticRegression(max_iter=1000)

penatly = ['l1', 'l2']
C= uniform(loc=0, scale=4).rvs(10)
print(C)

hyperparameters = dict(C=C)#, penatly=penatly)
print(hyperparameters)

randomizdsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0)
#randomizdsearch = RandomizedSearchCV(logistic, hyperparameters)

best_model = randomizdsearch.fit(features, target)

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# 载入数据
digits = load_digits()
X, y = digits.data, digits.target

# 建立一个分类器或者回归器
clf = RandomForestClassifier(n_estimators=20)

# 给定参数搜索范围：list or distribution
param_dist = {"max_depth": [3, None],                     #给定list
              "max_features": sp_randint(1, 11),          #给定distribution
              "min_samples_split": sp_randint(2, 11),     #给定distribution
              "bootstrap": [True, False],                 #给定list
              "criterion": ["gini", "entropy"]}           #给定list

# 用RandomSearch+CV选取超参数
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5)
random_search.fit(X, y)
