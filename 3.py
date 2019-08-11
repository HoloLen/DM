import numpy as np
import pandas as pd

pd.options.display.max_columns = 500
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.3, random_state=2018)

y_train = train_data.status.values
y_test = test_data.status.values

X_train = train_data.drop('status', axis = 1)
X_test = test_data.drop('status', axis = 1)
rom sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import roc_auc_score
print('auc:',roc_auc_score(y_test,y_pred))#随机森林

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
 
features_list = X_train.columns.values
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
 
plt.figure(figsize=(8,16))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()

select_fea = features_list[sorted_idx][10:]
X_train = X_train[select_fea]
X_test = X_test[select_fea]

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

from sklearn.metrics import roc_auc_score
print('auc:',roc_auc_score(y_test,y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
 
features_list = X_train.columns.values
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
 
plt.figure(figsize=(8,16))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importances')
plt.draw()
plt.show()


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.05, penalty='l1')
lr.fit(X_train_std, y_train)

from sklearn import SVM
from sklearn.svm import SVC
# 线性 SVM
linear_svc = SVC(kernel='linear', probability=True)
linear_svc.fit(X_train_std, y_train)
# 多项式 SVM
poly_svc = SVC(kernel='poly', probability=True)
poly_svc.fit(X_train_std, y_train)
