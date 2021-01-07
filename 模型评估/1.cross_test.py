from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

digits = datasets.load_digits()
features = digits.data
target = digits.target

standardizer = StandardScaler()
logit = LogisticRegression()
pipeline = make_pipeline(standardizer, logit)

kf = KFold(n_splits = 5, shuffle = True, random_state=1)

cv_results = cross_val_score(pipeline, features, target, cv = kf, scoring = 'accuracy',n_jobs=10)


print(cv_results.mean())
print(cv_results)

from sklearn.model_selection import train_test_split

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 1)
standardizer.fit(features_train)
features_train_std = standardizer.transform(features_train)
features_test_std = standardizer.transform(features_test)

cv_results = cross_val_score(pipeline, features, target, cv=kf, scoring='accuracy', n_jobs = -1)
print(cv_results)

