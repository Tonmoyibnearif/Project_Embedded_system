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
    regressor.add(LSTM( 50,return_sequences=True,input_shape=(1,9),recurrent_activation='hard_sigmoid',stateful=False))
    regressor.add(Dropout(0.1))
    regressor.add(BatchNormalization())
    regressor.add(LSTM( 50, return_sequences=True))
    regressor.add(Dropout(0.7))
    regressor.add(LSTM( 50, return_sequences=True))
    regressor.add(Dropout(0.8))
    regressor.add(Activation('relu'))
    regressor.add(LSTM( 50))
    regressor.add(Dropout(0.9))
    regressor.add(Activation('tanh'))
    regressor.add(Dense(1))
    regressor.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=[metrics.mse])
    return regressor
def Predictor(X_test,Y_test,Compiler_1):
    Predicted_windpower=Compiler_1.predict(X_test)
    MSE = Compiler_1.evaluate(X_test, Y_test, batch_size=100, verbose=0)
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

compiler_1=Build_model()
compiler_1.fit(X_train_1,Y_train_1,validation_data=(X_val_1,Y_val_1),epochs=10,batch_size=100)
compiler_2=Build_model()
compiler_2.fit(X_train_2,Y_train_2,validation_data=(X_val_2,Y_val_2),epochs=10,batch_size=100)
compiler_3=Build_model()
compiler_3.fit(X_train_3,Y_train_3,validation_data=(X_val_3,Y_val_3),epochs=10,batch_size=100)
compiler_4=Build_model()
compiler_4.fit(X_train_4,Y_train_4,validation_data=(X_val_4,Y_val_4),epochs=10,batch_size=100)
compiler_5=Build_model()
compiler_5.fit(X_train_5,Y_train_5,validation_data=(X_val_5,Y_val_5),epochs=10,batch_size=100)
compiler_6=Build_model()
compiler_6.fit(X_train_6,Y_train_6,validation_data=(X_val_6,Y_val_6),epochs=10,batch_size=100)
compiler_7=Build_model()
compiler_7.fit(X_train_7,Y_train_7,validation_data=(X_val_7,Y_val_7),epochs=10,batch_size=100)
compiler_8=Build_model()
compiler_8.fit(X_train_8,Y_train_8,validation_data=(X_val_8,Y_val_8),epochs=10,batch_size=100)
compiler_9=Build_model()
compiler_9.fit(X_train_9,Y_train_9,validation_data=(X_val_9,Y_val_9),epochs=10,batch_size=100)
compiler_10=Build_model()
compiler_1ß.fit(X_train_10,Y_train_10,validation_data=(X_val_10,Y_val_10),epochs=10,batch_size=100)
compiler_11=Build_model()
compiler_11.fit(X_train_11,Y_train_11,validation_data=(X_val_11,Y_val_11),epochs=10,batch_size=100)

Pred_1,Mse_1,Rmse_1=Predictor(X_test_1,Y_test_1,compiler_1)
Pred_2,Mse_2,Rmse_2=Predictor(X_test_2,Y_test_2,compiler_2)
Pred_3,Mse_3,Rmse_3=Predictor(X_test_3,Y_test_3,compiler_3)
Pred_4,Mse_4,Rmse_4=Predictor(X_test_4,Y_test_4,compiler_4)
Pred_5,Mse_5,Rmse_5=Predictor(X_test_5,Y_test_5,compiler_5)
Pred_6,Mse_6,Rmse_6=Predictor(X_test_6,Y_test_6,compiler_6)
Pred_7,Mse_7,Rmse_7=Predictor(X_test_7,Y_test_7,compiler_7)

print(Rmse_1)
print(Rmse_2)
print(Rmse_3)
print(Rmse_4)
print(Rmse_5)
print(Rmse_6)
print(Rmse_7)

visual=visualization(Y_test_7,Pred_7)





