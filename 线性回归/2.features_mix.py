from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

boston = load_boston()
features = boston.data[:0:2]
target = boston.target

interaction = PolynomialFeatures(degree = 3, include_bias = False, interaction_onlu = True)
features_interaction = interaction.fit_transform(features)

regression = LinearRegression()
model = regression.fit(features_interaction, target)

#7966
#6639


