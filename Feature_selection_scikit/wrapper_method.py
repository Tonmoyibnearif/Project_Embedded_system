from sklearn.linear_model import LinearRegression
import pandas as pd
dataset=pd.read_csv('DATA_WIth_DATE_data_Prtc_one.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
dataset.drop('Date', axis=1, inplace=True)
x = dataset.iloc[:, 0:19]
X_t_1=dataset.iloc[:-1, 0:19]
Y_tplus1=dataset.iloc[1:, 19]
X_t_2=dataset.iloc[:-2, 0:19]
Y_tplus2=dataset.iloc[2:, 19]
y = dataset.iloc[:, 19]
cols=x.columns.tolist()
lr=LinearRegression()
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
sfs = SFS(lr,
          k_features=10,
          forward=True,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=20)
#without Autoregression
sfs = sfs.fit(x, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
#Autoregression with time one hour
sfs_1= sfs.fit(X_t_1,Y_tplus1 )
fig = plot_sfs(sfs_1.get_metric_dict(), kind='std_err')
#Autoregression with time two hours
sfs_2= sfs.fit(X_t_2,Y_tplus2 )
fig = plot_sfs(sfs_2.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
print(sfs.k_feature_names_)
print(sfs.k_score_)
print(sfs.subsets_)

import pandas as pd
wrapper=pd.DataFrame.from_dict(sfs.get_metric_dict()).T
wrapper_1=pd.DataFrame.from_dict(sfs_1.get_metric_dict()).T
wrapper_2=pd.DataFrame.from_dict(sfs_2.get_metric_dict()).T

print('Selected features:', sfs.k_feature_idx_)
#print('Best features:', sfs.best_estimator_.steps[0][1].k_feature_idx_)
print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))


#with selected features
import datetime

t1 = datetime.datetime.now()
# DO CALCULATIONS HERE
X_selected=dataset.loc[:, ['AirPressure','cos_weekofyear','WindSpeed100m', 'WindSpeed10m', 'WindDirectionMeridional']]
sfs_S = SFS(lr,
          k_features=5,
          forward=True,
          floating=False,
          scoring='neg_mean_squared_error',
          cv=20)
sfs_selected = sfs_S.fit(X_selected, y)
fig = plot_sfs(sfs_selected.get_metric_dict(), kind='std_err')
t2 = datetime.datetime.now()
print("difference {}".format(t2 - t1))