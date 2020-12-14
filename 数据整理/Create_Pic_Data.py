import pandas as pd

dataframe = pd.DataFrame()

dataframe['Name'] = ['Jay', 'San']
dataframe['Age'] = [39, 24]
dataframe['Diver'] = [True, False]

print(dataframe)
new_person = pd.Series(['Mon', 40, True], index=['Name', 'Age', 'Diver'])
dataframe = dataframe.append(new_person, ignore_index=True)

print(dataframe.head(5))