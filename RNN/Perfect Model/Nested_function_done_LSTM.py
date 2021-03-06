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
    regressor.add(Dropout(0.1))
    regressor.add(BatchNormalization())
    regressor.add(LSTM( 50, return_sequences=True))
    regressor.add(Dropout(0.7))
    regressor.add(LSTM( 50, return_sequences=True))
    regressor.add(Dropout(0.8))
    regressor.add(Activation('tanh'))
    regressor.add(LSTM( 50))
    regressor.add(Dropout(0.9))
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
def visualization(Y_test_1,Pred_1):
    plt.figure(figsize=(10,8))
    plt.plot(Y_test_1, color='red',label='Test_Windpower_Data')
    plt.plot(Pred_1, color='blue',label='Predicted_Windpower_Data')
    plt.title('Windfarm_Power_prediction')
    plt.xlabel('Time')
    plt.xlabel('Power')
    plt.legend()
    plt.show()

#Variables Log
csv='Preprocessed_data_three.csv'
X_train_1,Y_train_1,X_val_1,Y_val_1,X_test_1,Y_test_1=Reshaper(csv,1)
X_train_2,Y_train_2,X_val_2,Y_val_2,X_test_2,Y_test_2=Reshaper(csv,2)
X_train_3,Y_train_3,X_val_3,Y_val_3,X_test_3,Y_test_3=Reshaper(csv,3)
X_train_4,Y_train_4,X_val_4,Y_val_4,X_test_4,Y_test_4=Reshaper(csv,4)
X_train_5,Y_train_5,X_val_5,Y_val_5,X_test_5,Y_test_5=Reshaper(csv,5)
X_train_6,Y_train_6,X_val_6,Y_val_6,X_test_6,Y_test_6=Reshaper(csv,6)
X_train_7,Y_train_7,X_val_7,Y_val_7,X_test_7,Y_test_7=Reshaper(csv,7)
X_train_8,Y_train_8,X_val_8,Y_val_8,X_test_8,Y_test_8=Reshaper(csv,8)
X_train_9,Y_train_9,X_val_9,Y_val_9,X_test_9,Y_test_9=Reshaper(csv,9)
X_train_10,Y_train_10,X_val_10,Y_val_10,X_test_10,Y_test_10=Reshaper(csv,10)
X_train_11,Y_train_11,X_val_11,Y_val_11,X_test_11,Y_test_11=Reshaper(csv,11)
X_train_12,Y_train_12,X_val_12,Y_val_12,X_test_12,Y_test_12=Reshaper(csv,12)
X_train_13,Y_train_13,X_val_13,Y_val_13,X_test_13,Y_test_13=Reshaper(csv,13)
X_train_14,Y_train_14,X_val_14,Y_val_14,X_test_14,Y_test_14=Reshaper(csv,14)
X_train_15,Y_train_15,X_val_15,Y_val_15,X_test_15,Y_test_15=Reshaper(csv,15)
X_train_16,Y_train_16,X_val_16,Y_val_16,X_test_16,Y_test_16=Reshaper(csv,16)
X_train_17,Y_train_17,X_val_17,Y_val_17,X_test_17,Y_test_17=Reshaper(csv,17)
X_train_18,Y_train_18,X_val_18,Y_val_18,X_test_18,Y_test_18=Reshaper(csv,18)
X_train_19,Y_train_19,X_val_19,Y_val_19,X_test_19,Y_test_19=Reshaper(csv,19)
X_train_20,Y_train_20,X_val_20,Y_val_20,X_test_20,Y_test_20=Reshaper(csv,20)
X_train_21,Y_train_21,X_val_21,Y_val_21,X_test_21,Y_test_21=Reshaper(csv,21)
X_train_22,Y_train_22,X_val_22,Y_val_22,X_test_22,Y_test_22=Reshaper(csv,22)
X_train_23,Y_train_23,X_val_23,Y_val_23,X_test_23,Y_test_23=Reshaper(csv,23)
X_train_24,Y_train_24,X_val_24,Y_val_24,X_test_24,Y_test_24=Reshaper(csv,24)
X_train_25,Y_train_25,X_val_25,Y_val_25,X_test_25,Y_test_25=Reshaper(csv,25)
X_train_26,Y_train_26,X_val_26,Y_val_26,X_test_26,Y_test_26=Reshaper(csv,26)
X_train_27,Y_train_27,X_val_27,Y_val_27,X_test_27,Y_test_27=Reshaper(csv,27)


#Predictor Log
Pred_1,Mse_1,Rmse_1=Predictor(X_train_1,Y_train_1,X_val_1,Y_val_1,X_test_1,Y_test_1)
Pred_2,Mse_2,Rmse_2=Predictor(X_train_2,Y_train_2,X_val_2,Y_val_2,X_test_2,Y_test_2)
Pred_3,Mse_3,Rmse_3=Predictor(X_train_3,Y_train_3,X_val_3,Y_val_3,X_test_3,Y_test_3)
Pred_4,Mse_4,Rmse_4=Predictor(X_train_4,Y_train_4,X_val_4,Y_val_4,X_test_4,Y_test_4)
Pred_5,Mse_5,Rmse_5=Predictor(X_train_5,Y_train_5,X_val_5,Y_val_5,X_test_5,Y_test_5)
Pred_6,Mse_6,Rmse_6=Predictor(X_train_6,Y_train_6,X_val_6,Y_val_6,X_test_6,Y_test_6)
Pred_7,Mse_7,Rmse_7=Predictor(X_train_7,Y_train_7,X_val_7,Y_val_7,X_test_7,Y_test_7)
Pred_8,Mse_8,Rmse_8=Predictor(X_train_8,Y_train_8,X_val_8,Y_val_8,X_test_8,Y_test_8)
Pred_9,Mse_9,Rmse_9=Predictor(X_train_9,Y_train_9,X_val_9,Y_val_9,X_test_9,Y_test_9)
Pred_10,Mse_10,Rmse_10=Predictor(X_train_10,Y_train_10,X_val_10,Y_val_10,X_test_10,Y_test_10)
Pred_11,Mse_11,Rmse_11=Predictor(X_train_11,Y_train_11,X_val_11,Y_val_11,X_test_11,Y_test_11)
Pred_12,Mse_12,Rmse_12=Predictor(X_train_12,Y_train_12,X_val_12,Y_val_12,X_test_12,Y_test_12)
Pred_13,Mse_13,Rmse_13=Predictor(X_train_13,Y_train_13,X_val_13,Y_val_13,X_test_13,Y_test_13)
Pred_14,Mse_14,Rmse_14=Predictor(X_train_14,Y_train_14,X_val_14,Y_val_14,X_test_14,Y_test_14)
Pred_15,Mse_15,Rmse_15=Predictor(X_train_15,Y_train_15,X_val_15,Y_val_15,X_test_15,Y_test_15)
Pred_16,Mse_16,Rmse_16=Predictor(X_train_16,Y_train_16,X_val_16,Y_val_16,X_test_16,Y_test_16)
Pred_17,Mse_17,Rmse_17=Predictor(X_train_17,Y_train_17,X_val_17,Y_val_17,X_test_17,Y_test_17)
Pred_18,Mse_18,Rmse_18=Predictor(X_train_18,Y_train_18,X_val_18,Y_val_18,X_test_18,Y_test_18)
Pred_19,Mse_19,Rmse_19=Predictor(X_train_19,Y_train_19,X_val_19,Y_val_19,X_test_19,Y_test_19)
Pred_20,Mse_20,Rmse_20=Predictor(X_train_20,Y_train_20,X_val_20,Y_val_20,X_test_20,Y_test_20)
Pred_21,Mse_21,Rmse_21=Predictor(X_train_21,Y_train_21,X_val_21,Y_val_21,X_test_21,Y_test_21)
Pred_22,Mse_22,Rmse_22=Predictor(X_train_22,Y_train_22,X_val_22,Y_val_22,X_test_22,Y_test_22)
Pred_23,Mse_23,Rmse_23=Predictor(X_train_23,Y_train_23,X_val_23,Y_val_23,X_test_23,Y_test_23)
Pred_24,Mse_24,Rmse_24=Predictor(X_train_24,Y_train_24,X_val_24,Y_val_24,X_test_24,Y_test_24)
Pred_25,Mse_25,Rmse_25=Predictor(X_train_25,Y_train_25,X_val_25,Y_val_25,X_test_25,Y_test_25)
Pred_26,Mse_26,Rmse_26=Predictor(X_train_26,Y_train_26,X_val_26,Y_val_26,X_test_26,Y_test_26)
Pred_27,Mse_27,Rmse_27=Predictor(X_train_27,Y_train_27,X_val_27,Y_val_27,X_test_27,Y_test_27)


from numpy import arange

Hour=[]
Hour=arange(27)

Ho=[[Rmse_1,Rmse_2,Rmse_3,Rmse_4,Rmse_5,Rmse_6,Rmse_7,Rmse_8,Rmse_9,Rmse_10,Rmse_11,Rmse_12,Rmse_13,Rmse_14,Rmse_15,Rmse_16,Rmse_17,Rmse_18,Rmse_19,Rmse_20,Rmse_21,Rmse_22,Rmse_23,Rmse_24,Rmse_25,Rmse_26,Rmse_27],[Mse_1,Mse_2,Mse_3,Mse_4,Mse_5,Mse_6,Mse_7,Mse_8,Mse_9,Mse_10,Mse_11,Mse_12,Mse_13,Mse_14,Mse_15,Mse_16,Mse_17,Mse_18,Mse_19,Mse_20,Mse_21,Mse_22,Mse_23,Mse_24,Mse_25,Mse_26,Mse_27]]


visual=visualization(Hour,Ho)


print(Mse_1)

plt.figure(figsize=(10,8))
plt.plot(Hour,Ho[0],'go-',label='RMSE')
plt.plot(Hour,Ho[1],'bo-',label='MSE')
#plt.plot(Pred_1, color='blue',label='Predicted_Windpower_Data')
plt.title('Windfarm_Power_Forecast_Error_Over_27_Hours_Autoregression')
plt.xlabel('Hours')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()