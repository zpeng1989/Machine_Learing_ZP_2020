from sklearn.feature_selection import VarianceThreshold

features = [[0,1,0],
            [0,1,1],
            [0,1,0],
            [0,1,1],
            [1,0,0]]

thresholder = VarianceThreshold(threshold=(0.75*(1-0.75)))
print(thresholder.fit_transform(features))
