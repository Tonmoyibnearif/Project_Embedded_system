from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from tensorflow.python.keras.optimizers import RMSprop
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import math
def load_data(CSV_path):
    dataset=pd.read_csv(CSV_path)
    dataset['Date']=pd.to_datetime(dataset['Date'])
    dataset=dataset.set_index(dataset['Date'])
    dataset=dataset.sort_index()
    return dataset
def Data_split(csv):
    data = load_data(csv)
    split_date = pd.datetime(2017,1,31)
    Test_date = pd.datetime(2017,2,28)
    Features=data.iloc[:, 11:46]
    df_training = Features.loc[data['Date'] <= split_date]
    df_test = Features.loc[data['Date'] > Test_date]
    val_mask=(data.Date > pd.to_datetime(split_date)) & (data.Date<=pd.to_datetime(Test_date))
    df_validation=Features.loc[val_mask]
    df_training = Features.loc[data['Date'] <= split_date]
    df_test = Features.loc[data['Date'] > split_date]
    return df_training,df_validation,df_test
def Autoregression(csv,window):
    Train,Validation,Test=Data_split(csv)
    X_train=Train.iloc[0:-window, 0:9].values
    Y_train=Train.iloc[window:, 9:].values
    X_test=Test.iloc[0:-window, 0:9].values
    Y_test=Test.iloc[window:, 9:].values
    X_val=Validation.iloc[0:-window, 0:9].values
    Y_val=Validation.iloc[window:, 9:].values
    return X_train,Y_train,X_val,Y_val,X_test,Y_test
def normalization(csv,window):
    X_train,Y_train,X_val,Y_val,X_test,Y_test=Autoregression(csv,window)
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_val=scaler.transform(X_val)
    X_test=scaler.transform(X_test)
    return  X_train,Y_train,X_val,Y_val,X_test,Y_test
def Reshaper(csv,window):
    X_train,Y_train,X_val,Y_val,X_test,Y_test=normalization(csv,window)
    X_train = X_train.reshape((-1,9,1))
    Y_train = Y_train.reshape((-1,1))
    X_val=X_val.reshape((-1,9,1))
    Y_val=Y_val.reshape((-1,1))
    X_test=X_test.reshape((-1,9,1))
    Y_test=Y_test.reshape((-1,1))
    return X_train,Y_train,X_val,Y_val,X_test,Y_test
from keras.models import Sequential
from keras.layers import LSTM,GRU,Dense,Dropout,Activation, Dense, BatchNormalization, TimeDistributed
from keras import metrics

def buildModel():
    model= Sequential()
    #model.add(Embedding(20000,32,input_length=X1))
    model.add(Conv1D(32,kernel_size=1,padding='same',activation='relu',input_shape=(X_train.shape[1],1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,kernel_size=2,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.35))
    model.add(Conv1D(128,kernel_size=2,padding='same',activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(GRU(50,return_sequences=True))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.45))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=[metrics.mse])
    return model
    #model.fit(X_train,Y_train,batch_size=100,epochs=10)
    #model.save("toxic.h5")
    #pred=model.predict(xtest)
    #return pred
def Predictor(X_train_x,Y_train_x,X_val_x,Y_val_x,X_test,Y_test):
    compiler_x=buildModel()
    compiler_x.fit(X_train_x,Y_train_x,validation_data=(X_val_x,Y_val_x),epochs=10,batch_size=100)
    Predicted_windpower=compiler_x.predict(X_test)
    MSE = compiler_x.evaluate(X_test, Y_test, batch_size=100, verbose=0)
    RMSE = math.sqrt(MSE[0])
    return Predicted_windpower,MSE[0],RMSE
def visualization(Hour,RMSError,MSError):
    plt.figure(figsize=(10,8))
    plt.plot(Hour,RMSError,'go-',label='RMSE')
    plt.plot(Hour,MSError,'bo-',label='MSE')
    #plt.plot(Pred_1, color='blue',label='Predicted_Windpower_Data')
    plt.title('Windfarm_Power_Forecast_Error_Over_27_Hours_Autoregression')
    plt.xlabel('Hours')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()
#Variables Log

csv='Preprocessed_data_three.csv'
RMSError=[]
MSError=[]
for i in range(1,27):
    X_train,Y_train,X_val,Y_val,X_test,Y_test=Reshaper(csv,i)
    Pred,Mse,Rmse=Predictor(X_train,Y_train,X_val,Y_val,X_test,Y_test)
    print("Runing Autoregression number:",i)
    RMSError.append(Rmse)
    MSError.append(Mse)


from numpy import arange

Hour=arange(26)



visual=visualization(Hour,RMSError,MSError)

