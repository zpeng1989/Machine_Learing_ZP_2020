import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

features, target = make_classification(n_samples=10000, n_features = 10, n_classes=2, n_informative=3, random_state=3)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.1, random_state=1)
logit = LogisticRegression()

logit.fit(features_train, target_train)

target_probabilites = logit.predict_proba(features_test)[:,1]
print(target_probabilites[:4])

false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilites)
plt.plot(false_positive_rate,true_positive_rate)
plt.show()