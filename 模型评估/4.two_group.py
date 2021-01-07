from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples = 1000, n_features = 3, n_informative = 3, n_redundant = 0, n_classes = 2, random_state = 1)

logit = LogisticRegression()

print(cross_val_score(logit, X, y, scoring = 'accuracy',cv=5))
print(cross_val_score(logit, X, y, scoring = 'recall', cv= 5))
print(cross_val_score(logit, X, y, scoring = 'f1'))