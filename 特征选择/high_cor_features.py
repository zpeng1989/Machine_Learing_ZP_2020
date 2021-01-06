import pandas as pd
import numpy as np

features = np.array([[1,1,1],[2,2,0],[3,3,1],[4,4,0],[5,5,1],[6,6,0],[7,7,1],[8,7,0],[9,7,1]])

dataframe = pd.DataFrame(features)

corr_matrix = dataframe.corr().abs()


