import pandas as pd
dataframe = pd.read_csv('titanic_dataset.csv')
print(dataframe.head(2))
print(dataframe.shape)
print(dataframe.describe())

print(dataframe.iloc[0])
print(dataframe.iloc[1:4])
dataframe = dataframe.set_index(dataframe['name'])
print(dataframe.head(2))
print(dataframe.loc['Allen, Miss. Elisabeth Walton'])

print(dataframe[dataframe['sex']=='female'].head(2))

print(dataframe[(dataframe['sex']=='female')&(dataframe['age']>=65)])

print(dataframe['sex'].replace('female', 'woman').head(2))


print(dataframe.rename(columns={'pclass':'passage'}).head(2))

print(dataframe['age'].mean())
print(dataframe['age'].count())
print(dataframe['sex'].count())

print(dataframe['sex'].unique())
print(dataframe['sex'].value_counts())


####################

print(dataframe.drop('age', axis=1).head())
print(dataframe[dataframe['sex']!='male'].head(2))
print(dataframe.drop_duplicates(subset=['sex'],keep='last'))


