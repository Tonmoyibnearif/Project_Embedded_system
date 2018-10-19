#import tensorflow
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
import matplotlib.pyplot as plt


dataset=pd.read_csv('Preprocessed_data_three.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
#Creating Spcial Dataset
#for power in range(1,2):
#    dataset['Power_'+str(power)]=dataset.PowerGeneration.shift(power)
cols=dataset.columns.tolist()
Features=dataset.iloc[:, 11:46]
Features=Features.fillna(0)
#Test and Train Spilt
split_date = pd.datetime(2017,1,31)
Test_date = pd.datetime(2017,2,28)
df_training = Features.loc[dataset['Date'] <= split_date]
df_test = Features.loc[dataset['Date'] > Test_date]
val_mask=(dataset.Date > pd.to_datetime(split_date)) & (dataset.Date<=pd.to_datetime(Test_date))
df_validation=Features.loc[val_mask]
df_training = Features.loc[dataset['Date'] <= split_date]
df_test = Features.loc[dataset['Date'] > split_date]


X_train=df_training.iloc[:, 0:9]
Y_train=df_training.iloc[:, 9:]
print(Y_train.shape)
X_test=df_test.iloc[:, 0:9]
Y_test=df_test.iloc[:, 9:]

X_val=df_validation.iloc[:, 0:9]
Y_val=df_validation.iloc[:, 9:]
#feature scalling
sc=MinMaxScaler(feature_range=(0,1))
X_train=sc.fit_transform(X_train)
#X_train=array(X_train)
#from numpy import array
#X_train=array(X_train)
#print(X_train.shape[0])

#Creating 2 Hour Autoregression
Testing_data=X_train.iloc[0:-2,0:6].values
Target_data=Y_train.iloc[2:,:].values
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
regressor.fit(tmp_x_train,tmp_y_train,epochs=100,batch_size=100)

Validation_data=X_test.reshape((-1,1,6))
Predicted_windpower=regressor.predict(Validation_data)

plt.figure(figsize=(15,10))
plt.plot(Y_test, color='red',label='Test_Windpower_Data')
plt.plot(Predicted_windpower, color='blue',label='Predicted_Windpower_Data')
plt.title('Windfarm_Power_prediction')
plt.xlabel('Time')
plt.xlabel('Power')
#plt.figure(figsize=(1,1))
plt.legend()
plt.show()

from sklearn import metrics
from sklearn.metrics import mean_squared_error
#MSE
MSE=(metrics.mean_squared_error(Y_test, Predicted_windpower))
#MAE
MAE=(metrics.mean_absolute_error(Y_test, Predicted_windpower))
#RMSe
RMSE=(np.sqrt(metrics.mean_squared_error(Y_test, Predicted_windpower)))
#to print numpy array
np.set_printoptions(precision=15)
print ("MAE:%2f" %(MAE))
print ("MSE:%2f" %(MSE))
print ("RMSE:%2f" %(RMSE))


