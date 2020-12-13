from sklearn import datasets

digits = datasets.load_digits()

features = digits.data
target = digits.target

print(features[0])
print(features[0].shape)
print(target[0])
