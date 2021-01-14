from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree

booston = datasets.load_boston()
features = booston.data[:,0:2]
target = booston.target

decisiontree = DecisionTreeRegressor(random_state=0)
model = decisiontree.fit(features, target)

observations = [[0.02,16]]
print(model.predict(observations))

iris = datasets.load_iris()
features = iris.data
target = iris.target

decisiontree = DecisionTreeClassifier(random_state=0)
model = decisiontree.fit(features, target)

dot_data = tree.export_graphviz(decisiontree, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())
