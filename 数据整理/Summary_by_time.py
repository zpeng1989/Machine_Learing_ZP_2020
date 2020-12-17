import pandas as pd
import numpy as np

time_index = pd.date_range('06/06/2017', periods=1000000, freq='30s')

dataframe = pd.DataFrame(index = time_index)

dataframe['Sale_Amount'] = np.random.randint(1,10, 1000000)
print(dataframe.head)
print(dataframe.resample('M').sum())