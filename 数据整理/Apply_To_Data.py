import pandas as pd

dataframe = pd.read_csv('titanic_dataset.csv')

def uppercase(x):
    return(x.upper())

print(dataframe['name'].apply(uppercase))

print(dataframe.groupby('sex').apply(lambda x: x.count()))

data_a = {'id': ['1', '1', '2'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}

dataframe_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])


print(dataframe_a)


data_b = {'id': ['4','5','6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns=['id', 'first', 'last'])

print(pd.concat([dataframe_a, dataframe_b], axis = 0))
print(pd.concat([dataframe_a, dataframe_b], axis = 1))

##### merge

employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees', 'Tim Horton']}

dataframe_employees = pd.DataFrame(employee_data, columns=['employee_id', 'name'])
print(dataframe_employees)

sales_data = {'employee_id': ['3', '4', '5'],
              'total_sales':[234234,53223,23423]}
dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id', 'total_sales'])
print(dataframe_sales)

print(pd.merge(dataframe_employees, dataframe_sales, on= 'employee_id'))
print(pd.merge(dataframe_employees, dataframe_sales, on= 'employee_id', how = 'outer'))
print(pd.merge(dataframe_employees, dataframe_sales, left_on= 'employee_id', right_on = 'employee_id'))






