import pandas as pd
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
    X_train = X_train.reshape((-1,1,9))
    Y_train = Y_train.reshape((-1,1))
    X_val=X_val.reshape((-1,1,9))
    Y_val=Y_val.reshape((-1,1))
    X_test=X_test.reshape((-1,1,9))
    Y_test=Y_test.reshape((-1,1))
    return X_train,Y_train,X_val,Y_val,X_test,Y_test
#LSTM Building Block
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Activation, Dense, BatchNormalization, TimeDistributed
from keras import metrics
def Build_model():
    regressor=Sequential()
    regressor.add(LSTM( 50,return_sequences=True,input_shape=(1,9)))
    #regressor.add(Dropout(0.1))
    regressor.add(BatchNormalization())
    regressor.add(LSTM( 50, return_sequences=True))
    #regressor.add(Dropout(0.7))
    regressor.add(LSTM( 50, return_sequences=True))
    #regressor.add(Dropout(0.8))
    regressor.add(Activation('tanh'))
    regressor.add(LSTM( 50))
    #regressor.add(Dropout(0.9))
    regressor.add(Dense(1))
    regressor.add(Activation('relu'))
    regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=[metrics.mse])
    return regressor
def Predictor(X_train_x,Y_train_x,X_val_x,Y_val_x,X_test,Y_test):
    compiler_x=Build_model()
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
    plt.title('Windfarm_Power_Forecast_Error_Over_27_Hours_Autoregression_Without_Dropout')
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



