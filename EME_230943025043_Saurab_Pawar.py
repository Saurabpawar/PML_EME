# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 08:11:57 2024

@author: Saurab
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
#load dataset

os.chdir("D:\CDAC DBDA")
hr = pd.read_csv("HR_comma_sep.csv")
#Encoding using hot encoding get dummies
dum_hr = pd.get_dummies(hr, drop_first=True)

X = dum_hr.drop('left', axis=1)
y = dum_hr['left']
#split train test for splitting data 
X_train, X_test, y_train, y_test = train_test_split(X,y, 
                               test_size=0.3,
                               stratify=y,
                               random_state=2022)
#Kfold with 5 folds
kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=2022)

#---------------------------------------------------------------------------------------------
#Question 1
#A

gn=GaussianNB()
#Default parameters
params={} 
#Grid search Cv for parameter tuning with neg loss and default params
gcv=GridSearchCV(gn,param_grid=params,cv=kfold,scoring='neg_log_loss')
#fitting model
gcv.fit(X,y)
#best accuracy with neg_log_loss
print(gcv.best_score_) 
#result
#-0.7181290454863862

#----------------------------------------------------------------------------------------------
#B
#logistic regression as data is categorical
lr=LogisticRegression()
#Use grid search for tunning model with default params and neg_log_loss
gcv=GridSearchCV(lr,param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X,y)
#best accuracy
print(gcv.best_score_)
#best accuracy with neg_log_loss
#-0.4326627613536142

#--------------------------------------------------------------------------------------------
#C
#random forest classifier as data is categorical
rf = RandomForestClassifier(random_state=2022,
                            n_estimators=20)
#params given 
params = {'max_features':[3,5,6],
          'max_depth': [None,3,6]}

#using grid search for tunnig model with neg_log_loss
gcv = GridSearchCV(rf, param_grid=params,verbose=2,
                   cv=kfold, scoring='neg_log_loss')
#fitting model
gcv.fit(X, y)
#best accuracay

print(gcv.best_score_)

#-0.11600062711097978

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Answer = RANDOM FOREST IS BEST FIT FOR THIS DATA AS ITS ACCURACY SCORE WITH LOG_LOSS IS -0.11073





################################################################################################


#Question 2
#A

nutri = pd.read_csv("nutrient.csv",index_col=0)
#Using standard scaler
scaler = StandardScaler()
nutriscaled=scaler.fit_transform(nutri)

#Using loop to identify best cluster 
Ks = [3,4,5,6,7,8,9,10]
scores = []
for k in Ks:
    clust = KMeans(n_clusters=k)
    clust.fit(nutriscaled)
    scores.append(silhouette_score(nutriscaled, clust.labels_))    

i_max = np.argmax(scores)
best_k = Ks[i_max]
#Best accuracy
print("Best Score:", scores[i_max])
#Best Cluster
print("Best No. of Clusters:", best_k)
#---------------------------------------------------------------
#line plot
plt.scatter(Ks, scores, c='red')
plt.plot(Ks, scores)
plt.xlabel("No. of CLusters")
plt.ylabel("WSS")
plt.title("Scree Plot")
plt.show()

#Best Score: 0.41997441967765275
#Best No. of Clusters: 4




#------------------------------------------------------------------------------------------------

#Question 2
#B

# dataset
df = pd.read_csv('nutrient.csv')

# Removed'Food_Item'
X = df.drop('Food_Item', axis=1)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#number of clusters 
n_clusters_list = [3, 4, 5, 6, 7, 8, 9, 10]

# silhouette scores
silhouette_scores = []

# Perform K-means clustering 
for n_clusters in n_clusters_list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=2022)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Generate a line plot 
plt.plot(n_clusters_list, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Clusters')
plt.show()

#PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Find the number of principal components for 70% variation
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components_70_percent = np.argmax(cumulative_variance_ratio >= 0.7) + 1

print("Number of Principal Components :", num_components_70_percent)

#Answer
#Number of Principal Components are : 3








#-------------------------------WITH PIPELINE----------------------------------------------------------
#standard scaling
scaler = StandardScaler()
prcomp = PCA()

#Pipline performing standard scaling and PCA
pipe = Pipeline([('SCL',scaler),('PCA',prcomp)])
comps = pipe.fit_transform(nutri)

print("Variance Ratios : ",prcomp.explained_variance_ratio_)

cum_sum = np.cumsum(prcomp.explained_variance_ratio_*100)
print("\nVariance Ratios percentage : ",cum_sum)

#As per observation Number of Principal Components are : 3