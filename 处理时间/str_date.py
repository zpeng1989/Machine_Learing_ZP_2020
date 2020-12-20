import numpy as np
import pandas as pd

date_strings = np.array(['03-04-2005 11:35 PM', '23-05-2010 12:01 AM', '04-09-2009 09:09 PM'])

print([pd.to_datetime(date, format= '%d-%m-%Y %I:%M %p') for date in date_strings])
print(pd.Timestamp('2017-05-01 06:00:00', tz = 'Europe/London').tz_convert('Africa/Abidjan'))

dataframe = pd.DataFrame()

dataframe['date'] = pd.date_range('1/1/2001', periods=1000000, freq= 'H')

print(dataframe[(dataframe['date']>'2002-1-1 01:00:00')&(dataframe['date']<= '2002-1-1 04:00:00')])

dataframe = dataframe.set_index(dataframe['date'])

print(dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00'])

dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
print(dataframe.head())

dataframe = pd.DataFrame()

dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

print(dataframe['Left']-dataframe['Arrived'])

dates = pd.Series(pd.date_range("2/2/2002", periods = 3, freq="M"))

#print(dates.dt.weekday_name)
print(dates.dt.weekday)


