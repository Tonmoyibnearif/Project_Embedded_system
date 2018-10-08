import numpy as np
import tensorflow
import pandas as pd
import seaborn as sns
import matplotlib
from sklearn.feature_selection import mutual_info_regression,f_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# allow plots to appear directly in the notebook
%matplotlib inline
dataset=pd.read_csv('DATA_WIth_DATE_data_Prtc_one.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
dataset.drop('Date', axis=1, inplace=True)
x = dataset.iloc[:-2, 0:19]
y = dataset.iloc[2:, 19]
cols=x.columns.tolist()

#f_regression feature selection
features=f_regression(x,y,center=True)
list_f=list(features)

print(list_f[0])
#mutual_info_regression feature selection
feature=mutual_info_regression(x,y,discrete_features='auto',n_neighbors=3,copy=True,random_state=None)
#feature selection using linear regression
lr = LinearRegression()
lr.fit(x, y)
coefficient=lr.coef_
print(lr.intercept_)
lr=lr.coef_

#passing all values into objects
feature_linear=list(zip(cols,lm2.coef_))
feature_Mutual=list(zip(cols,feature))
feature_f_reg1=list(zip(cols,list_f[0]))
feature_f_reg2=list(zip(cols,list_f[1]))


# visualize the relationship between the features and the response using scatterplots
sns.pairplot(dataset, x_vars=['cos_monthofyear','sin_monthofyear','AirPressure','Temperature','Humidity','WindSpeed100m','WindSpeed10m','WindDirectionZonal','WindDirectionMeridional'], y_vars='PowerGeneration', size=7, aspect=0.7)
#plotting bar graph to visulize the data
plt.bar(cols,list_f[0],label='f_regression',color='r')
plt.bar(cols,list_f[1],label='f_regression',color='b')
plt.bar(cols,feature,label='Mutual_regression',color='c')
plt.bar(cols,lr,label='linear_regression',color='c')
#bar plot with angle Mutual Info regression
X_selected=dataset.loc[:, ['AirPressure','cos_weekofyear','WindSpeed100m', 'WindSpeed10m', 'WindDirectionMeridional']]
feature_s=mutual_info_regression(X_selected,y,discrete_features='auto',n_neighbors=3,copy=True,random_state=None)
colss=X_selected.columns.tolist()
X_selected = range(len(colss))
plt.xticks(X_selected,  colss)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.plot(X_selected, feature_s)
#bar plot with  linearregression
x = range(len(cols))
plt.xticks(x,  cols)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.plot(x, lr)
#bar plot with angle f_regression F_values
x = range(len(cols))
plt.xticks(x,  cols)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.plot(x, list_f[0])
#bar plot with angle f_regression P_values
x = range(len(cols))
plt.xticks(x,  cols)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.plot(x, list_f[1])