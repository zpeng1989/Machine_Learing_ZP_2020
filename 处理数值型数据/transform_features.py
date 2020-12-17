import numpy as np
import pandas as pd

from sklearn.preprocessing import FunctionTransformer

features = np.array([[2,3], [2,3], [2,3]])

def add_ten(x):
    return x + 10

ten_transformer = FunctionTransformer(add_ten)
print(ten_transformer.transform(features))

df = pd.DataFrame(features, columns=['feature_1', 'feature_2'])
print(df.apply(add_ten))

