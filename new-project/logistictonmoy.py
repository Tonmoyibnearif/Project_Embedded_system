import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Social_Network_Ads.csv" )
X=dataset.iloc[:, [2,3]].values
Y=dataset.iloc[:, 4].values
#missing data handlelling
#splitting the datza into trianing ste and test set
from sklearn.cross_validation import train_test_split
X_train, X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
#creating object
classifier= LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)
#predicting the Test set results
y_pred=classifier.predict(X_test)
#confussion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,y_pred)
import keras