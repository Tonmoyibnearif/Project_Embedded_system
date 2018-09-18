# To avoid warning from python console
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
#x= input ("Enter csv name")
    def CSV_preprocessing(model):
         df = pd.read_csv(model)
         tom = pd.date_range('2015-07-03', periods=15927, freq='H')
         df['Date'] = [d.date() for d in tom]
         tumi = df['Date']
         tumi = pd.Series(pd.to_datetime(tumi))
         day_of_the_year = tumi.dt.dayofyear
         day_of_the_month = tumi.dt.day
         day_of_the_week = tumi.dt.dayofweek
         week_of_the_year = tumi.dt.weekofyear
         month_of_the_year = tumi.dt.month
    # replacing all those columns in the dataframe
         df.insert(0, 0, day_of_the_year)
         df.insert(0, 1, day_of_the_month)
         df.insert(0, 2, day_of_the_week)
         df.insert(0, 3, week_of_the_year)
         df.insert(0, 4, month_of_the_year)
    # cols= df.columns.tolist()
    # cols.insert(0,cols.pop(-1))
        # df=df[cols]
        # df['Time']=[d.time() for d in tom ]
         df['Monthofyear'] = df[4]
         cols = df.columns.tolist()
         cols.insert(0, cols.pop(-1))
         df = df[cols]
        # dropping a column
         df = df.drop(df.columns[1], axis=1)
         df['Weekofyear'] = df[3]
         cols = df.columns.tolist()
         cols.insert(1, cols.pop(-1))
         df = df[cols]
         df = df.drop(df.columns[2], axis=1)
         df['Dayofweek'] = df[2]
         cols = df.columns.tolist()
         cols.insert(2, cols.pop(-1))
         df = df[cols]
         df = df.drop(df.columns[3], axis=1)
         df['Dayofmonth'] = df[1]
         cols = df.columns.tolist()
         cols.insert(3, cols.pop(-1))
         df = df[cols]
         df = df.drop(df.columns[4], axis=1)
         df['Dayofyear'] = df[0]
         cols = df.columns.tolist()
         cols.insert(4, cols.pop(-1))
         df = df[cols]
         df = df.drop(df.columns[5], axis=1)
        # replacing date column
         cols = df.columns.tolist()
         cols.insert(0, cols.pop(-1))
         df = df[cols]
        # printing colums with object type
         colss = df.columns[df.dtypes.eq(object)]
         print(colss)
        # generating one hour time split
         tom = pd.Series(pd.to_datetime(tom))
         Hours = tom.dt.hour
         df['Time'] = Hours
        # creating sin and cos transfromation
         hours_in_day = 24
         df['sin_time'] = np.sin(2 * np.pi * df.Time / hours_in_day)
         df['cos_time'] = np.cos(2 * np.pi * df.Time / hours_in_day)
        # replacing the sine and cos time in the dataframe
        # cos_time replaced
         cols = df.columns.tolist()
         cols.insert(7, cols.pop(-1))
         df = df[cols]
    # sine_time replaced
         cols = df.columns.tolist()
         cols.insert(8, cols.pop(-1))
         df = df[cols]
        # handling dayof year
         days_in_year = 365
         df['sin_dayofyear'] = np.sin(2 * np.pi * df.Dayofyear / days_in_year)
         df['cos_dayof year'] = np.cos(2 * np.pi * df.Dayofyear / days_in_year)
        # sin and cos replaCING
         cols = df.columns.tolist()
         cols.insert(6, cols.pop(-1))
         cols.insert(7, cols.pop(-1))
         df = df[cols]
        # handling day of month
         days_in_month = 30
         df['sin_dayofmonth'] = np.sin(2 * np.pi * df.Dayofmonth / days_in_month)
         df['cos_dayofmonth'] = np.cos(2 * np.pi * df.Dayofmonth / days_in_month)
        # sin and cos replaCING
         cols = df.columns.tolist()
         cols.insert(5, cols.pop(-1))
         cols.insert(6, cols.pop(-1))
         df = df[cols]
        # handling day of week
         days_in_week = 7
         df['sin_dayofweek'] = np.sin(2 * np.pi * df.Dayofweek / days_in_week)
         df['cos_dayofweek'] = np.cos(2 * np.pi * df.Dayofweek / days_in_week)
        # sin and cos replaCING
         cols = df.columns.tolist()
         cols.insert(4, cols.pop(-1))
         cols.insert(5, cols.pop(-1))
         df = df[cols]
        # handling week of year
         weeks_in_year = 52
         df['sin_weekofyear'] = np.sin(2 * np.pi * df.Weekofyear / weeks_in_year)
         df['cos_weekofyear'] = np.cos(2 * np.pi * df.Weekofyear / weeks_in_year)
        # sin and cos replaCING
         cols = df.columns.tolist()
         cols.insert(3, cols.pop(-1))
         cols.insert(4, cols.pop(-1))
         df = df[cols]
        # handling month of year
         months_in_year = 52
         df['sin_monthofyear'] = np.sin(2 * np.pi * df.Monthofyear / months_in_year)
         df['cos_monthofyear'] = np.cos(2 * np.pi * df.Monthofyear / months_in_year)
        # sin and cos replaCING
         cols = df.columns.tolist()
         cols.insert(2, cols.pop(-1))
         cols.insert(3, cols.pop(-1))
         df = df[cols]
    # now dropping off unneccessary columns
         df.drop('Time', axis=1, inplace=True)
         df.drop('Date', axis=1, inplace=True)
         df.drop('Dayofweek', axis=1, inplace=True)
         df.drop('Monthofyear', axis=1, inplace=True)
         df.drop('Weekofyear', axis=1, inplace=True)
         df.drop('Dayofyear', axis=1, inplace=True)
         df.drop('Dayofmonth', axis=1, inplace=True)
        # dropping the Forecasting Time column from dataframe
         df.drop('ForecastingTime', axis=1, inplace=True)
#        #saving the file in csv
#    #https://www.youtube.com/watch?v=-0NwrcZOKhQ
         df.to_csv('Preprocessed_data_three.csv',index=False)
#         # implementing ANN
#         # spliting data into x-independent and y-dependent variable
#         x = df.iloc[:, :19].values
#         y = df.iloc[:, 19].values
         return df,

#mydata=CSV_preprocessing("wf3.csv")

#dff = pd.read_csv("Preprocessed_data_two.csv")
