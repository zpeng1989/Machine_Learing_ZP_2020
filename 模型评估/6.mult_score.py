from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

features, target = make_classification(n_samples = 10000,
                                      n_features = 3,
                                      n_informative=3,
                                      n_redundant = 0,
                                      n_classes = 3,
                                      random_state = 1)


logit = LogisticRegression()


print(cross_val_score(logit, features, target, scoring = 'accuracy',cv =3))

print(cross_val_score(logit, features, target, scoring = 'f1_macro',cv =3)
      
      )
