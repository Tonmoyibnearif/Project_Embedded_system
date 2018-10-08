import tensorflow
import pandas as pd
import os
path="./hdf5/"
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from tensorflow.python.keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.layers import LSTM


dataset=pd.read_csv('Preprocessed_data_three.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
#dataset.drop('Date', axis=1, inplace=True)
#splitting the Data into test and train
features=dataset.loc[:, ['Date','cos_weekofyear', 'sin_time', 'AirPressure', 'WindSpeed100m', 'WindDirectionZonal', 'WindDirectionMeridional','PowerGeneration']]

split_date = pd.datetime(2017,3,31)

df_training = features.loc[dataset['Date'] <= split_date]
df_test = features.loc[dataset['Date'] > split_date]

df_test.drop('Date', axis=1, inplace=True)
df_training.drop('Date', axis=1, inplace=True)

df_test=df_test.reset_index()
df_training=df_training.reset_index()
X_train=df_training.iloc[:, 1:7]
Y_train=df_training.iloc[:, 7:]
print(Y_train.shape)
X_test=df_test.iloc[:, 1:7].values
Y_test=df_test.iloc[:, 7:].values
#feature scalling
sc=MinMaxScaler(feature_range=(0,1))
X_train=sc.fit_transform(X_train)
#X_train=array(X_train)
from numpy import array
X_train=array(X_train)
print(X_train.shape[0])

#Creating 2 Hour Autoregression
Testing_data=X_train.iloc[0:-2,0:6].values
Target_data=Y_train.iloc[2:0,0:].values
print(Testing_data.shape)

Testing_data=np.array(Testing_data)
#Henze Code
tmp_x=Testing_data[0:15310, :]
tmp_y=Target_data[0:15310, :]
#tmp_x.reshape((-1,24,6))
#tmp_x_train = tmp_x.reshape((-1,24,6))
#tmp_y_train = tmp_y.reshape((-1,24,1))
#regressor.fit(tmp_x_train,tmp_y_train)
tmp_x_train = tmp_x.reshape((-1,1,6))
#tmp_y_train = tmp_y.reshape((-1,1,1))
tmp_y_train = tmp_y.reshape((-1,1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
#RNN initialization
regressor=Sequential()
#adding first LSTM layer and dropour
regressor.add(LSTM( 50, return_sequences=True,input_shape=(1,6)))
regressor.add(Dropout(0.2))
#adding secong LSTM layer and dropout
regressor.add(LSTM( 50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding Third LSTM layer and dropout
regressor.add(LSTM( 50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding Fourth LSTM layer and dropout
regressor.add(LSTM( 50))
regressor.add(Dropout(0.2))
#Adding the output layer
#adding Third LSTM layer and dropout
regressor.add(Dense(1))
#compling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#fitting the RNN to Training set
regressor.fit(tmp_x_train,tmp_y_train,epochs=10,batch_size=100)



