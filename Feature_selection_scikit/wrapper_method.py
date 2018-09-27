from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('DATA_WIth_DATE_data_Prtc_one.csv')
dataset['Date']=pd.to_datetime(dataset['Date'])
dataset=dataset.set_index(dataset['Date'])
dataset=dataset.sort_index()
dataset.drop('Date', axis=1, inplace=True)
x = dataset.iloc[:, 0:19]
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

sfs = sfs.fit(X_t_2, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
print(sfs.k_feature_names_)
print(sfs.k_score_)
print(sfs.subsets_)

import pandas as pd
wrapper=pd.DataFrame.from_dict(sfs.get_metric_dict()).T

print('Selected features:', sfs.k_feature_idx_)
