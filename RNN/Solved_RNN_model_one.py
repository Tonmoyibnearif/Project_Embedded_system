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
#from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint.TensorBoard,ReduceLRoPLateau
#importing data using Pandas
#making Datatime columns and putting in Index
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
Y_train=df_training.iloc[:, 7:].values
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

Testing_data=X_train.iloc[0:-2,0:6].values

nb_samples = Testing_data.shape[0] - 2

Testing_data=np.array(Testing_data)
#Henze Code
tmp_x=Testing_data[0:15288, :]
tmp_y=Y_train[0:15288, :]
tmp_x.reshape((-1,24,6))
tmp_x_train = tmp_x.reshape((-1,24,6))
tmp_y_train = tmp_y.reshape((-1,24,1))
regressor.fit(tmp_x_train,tmp_y_train)
tmp_x_train = tmp_x.reshape((-1,1,6))
tmp_y_train = tmp_y.reshape((-1,1,1))
tmp_y_train = tmp_y.reshape((-1,1))


#esting_data=np.reshape(Testing_data(Testing_data.shape[0],2,6))
#+sting_data=np.reshape(Testing_data,(Testing_data.shape[0],2,6))

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

#Testing The LSTM
Dataset_total=pd.concat((train['PowerGeneration'],test['PowerGeneration']),axis=0)
inputs =Dataset_total[len(Dataset_total)-len(test)-60:].values
print(len(test))
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
Testing_data=[]
for i in range(10,828):
    Testing_data.append(inputs[i-60:i, 0])
Testing_data=np.array(Testing_data)
Testing_data=np.reshape(Testing_data,(Testing_data.shape[0],1,Testing_data.shape[1]))
Predicted_windpower=regressor.predict(Testing_data)
Predicted_windpower=sc.inverse_transform(Predicted_windpower)
#Visulization
plt.plot(test, color='red',label='Test_Windpower_Data')
plt.plot(Predicted_windpower, color='blue',label='Predicted_Windpower_Data')
plt.title('Windfarm_Power_prediction')
plt.xlabel('Time')
plt.xlabel('Power')
plt.legend()
plt.show()


# Saving Entire model to HDF5
model.save(os.path.join(path,"RNNmodel.h5"))
print("Saved model to disk")
# from keras.models import load_model
# model2=load_model(os.path.join(path,"RNNmodel.h5"))
# y_pred=model2.predict(x_test)
#
