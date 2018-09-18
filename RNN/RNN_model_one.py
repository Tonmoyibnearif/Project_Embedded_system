import tensorflow
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.layers import LSTM
#from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint.TensorBoard,ReduceLRoPLateau
#importing data using Pandas
#making Datatime columns and putting in Index
dataset=pd.read_csv('DATA_WIth_DATE_data_Prtc_one.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
dataset.drop('Date', axis=1, inplace=True)
#splitting the Data into test and train
train=dataset['2015-07-03':'2017-05-05']
test=dataset['2017-05-06':]
#make sure have numpy columns
Dataset_numpy=train.iloc[:, 0:20].values
#feature scalling
sc=MinMaxScaler(feature_range=(0,1))
Dataset_scalled=sc.fit_transform(Dataset_numpy)
#
#X=Dataset_scalled[:, 0:19]
#Y=Dataset_scalled[:, 19]
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

#Creatinga data structure with Timesteps
#splitting the data set
x_train=[]
y_train=[]
for i in range(60,16152):
    x_train.append(Dataset_scalled[i-60:i, 19])
    y_train.append(Dataset_scalled[i, 19])
x_train,y_train=np.array(x_train),np.array(y_train)
#reshape
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
#RNN initialization
regressor=Sequential()
#adding first LSTM layer and dropour
regressor.add(LSTM( units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))
#adding secong LSTM layer and dropout
regressor.add(LSTM( units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding Third LSTM layer and dropout
regressor.add(LSTM( units=50, return_sequences=True))
regressor.add(Dropout(0.2))
#adding Fourth LSTM layer and dropout
regressor.add(LSTM( units=50))
regressor.add(Dropout(0.2))
#Adding the output layer
#adding Third LSTM layer and dropout
regressor.add(Dense( units=1))
#compling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#fitting the RNN to Training set
regressor.fit(x_train,y_train,epochs=100,batch_size=32)



