import pandas as pd
import numpy as np


dataframe = pd.read_csv('titanic_dataset.csv')

print(dataframe.groupby('sex').mean())

print(dataframe.groupby('survived')['name'].count())
print(dataframe.columns)
print(dataframe.groupby(['sex', 'survived'])['name'].count())
print(dataframe.groupby(['sex', 'survived'])['age'].mean())
