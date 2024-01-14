# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:12:06 2023

@author: Saurab
"""

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as mse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans



#Q1------------------------------------------------------------------------------------------

#A----------------------------------------------------------------------------------------

os.chdir("D:\CDAC DBDA\Advance Analytics\Datasets")
sac= pd.read_csv("sacremento.csv",index_col=0)
dum_sac = pd.get_dummies(sac, drop_first=True)

X=dum_sac.drop(['price'],axis=1)
y=dum_sac['price']
kfold=KFold(n_splits=5,shuffle=True,random_state=23)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=23)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lasso = Lasso()
parameters = {'alpha': [0, 0.01, 1,1.5,2]}

gcv = GridSearchCV(lasso, parameters, cv=kfold,scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



#Q1-------------------------------------------------------------------

#B-------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=23,
                            n_estimators=20)

params = {'max_features':[2,4,6,8,10],
          'min_samples_split':[2, 5, 20, 80, 100],
          'max_depth': [3,4,6,7,None],
          'min_samples_leaf':[1, 5, 10, 20]}

gcv = GridSearchCV(rf, param_grid=params,
                   cv=kfold, scoring='r2',n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)




#Q2----------------------------------------------------------------------------------------------------


#A---------------------------------------------------------------------------------
US = pd.read_csv("USArrests.csv",index_col=0)

scaler=StandardScaler()
USscaled=scaler.fit_transform(US)

ks=[3,4,5,6,7,8,9,10]
scores=[]
for k in ks:
    clust=KMeans(n_clusters=k)
    clust.fit(USscaled)
    scores.append(silhouette_score(USscaled,clust.labels_))
    
i_max=np.argmax(scores)
best_k=ks[i_max]
print("Best Score:",scores[i_max])
print("Best No.of Clusters:",best_k)

plt.scatter(ks,scores,c='red')
plt.plot(ks,scores)
plt.xlabel("Nor of clusters")
plt.ylabel('WSS')
plt.title('Screen plot')
plt.show()

#B-----------------------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
prcomp = PCA()
lr = LogisticRegression()
prcomp = PCA(n_components=0.70)
scaler = StandardScaler()
m_scaled = scaler.fit_transform(US)

prcomp = PCA()
comps = prcomp.fit_transform(m_scaled)
print(US.shape)
print(comps.shape)

df_comps = pd.DataFrame(comps,
                        columns=['PC1','PC2',
                                 'PC3','PC4'],
                        index=US.index)

print(df_comps.var())
print(prcomp.explained_variance_)
tot_var = np.sum(prcomp.explained_variance_)
prop_var = np.array(prcomp.explained_variance_)/tot_var
print(prop_var)
per_var = prop_var*100
print(per_var)
print("%age var explained by 1st two PCs:",77.57590469+17.74794969)
print(prcomp.explained_variance_ratio_)
