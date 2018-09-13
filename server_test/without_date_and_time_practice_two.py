
import pandas as pd
import numpy as np

df = pd.read_csv('Preprocessed_data_Prtc_one.csv')
cols = df.columns.tolist()
x = df.iloc[:, 12:19].values
y = df.iloc[:, 19].values


# spliting the data into test and trian
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)



import keras
# make sequence of layers
from keras.models import Sequential
# using Dense function we can randomly intialize the weight
from keras.layers import Dense
from sklearn import metrics

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def baseline_model(model):

    model.fit(x_train,y_train,epochs=160,batch_size=100,verbose=0)
    return model

#estimator=KerasRegressor(build_fn=baseline_model,epochs=100,batch_size=10,verbose=0)
#results=cross_val_score(estimator,x_train,y_train)
#print("results: %2f (%.2f)MSE" % (results.mean(),results.std()))

#Evaluation of Neural network
model= Sequential()
model.add(Dense(10,input_dim=7,kernel_initializer='normal',activation='relu'))
model.add(Dense(1,kernel_initializer='normal'))
model.compile(loss='mean_squared_error',optimizer='adam')
for i in range(10):
 model = baseline_model(model)


y_pred=model.predict(x_test)
#MSE
MSE=(metrics.mean_squared_error(y_test, y_pred))
#MAE
MAE=(metrics.mean_absolute_error(y_test, y_pred))
#RMSe
RMSE=(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#to print numpy array
np.set_printoptions(precision=15)
print ("MAE:%2f" %(MAE))
print ("MSE:%2f" %(MSE))
print ("RMSE:%2f" %(RMSE))
#hi