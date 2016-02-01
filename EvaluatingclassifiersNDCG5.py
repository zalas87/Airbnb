# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:09:59 2016

@author: gonzalo
"""

import pandas as pd
import numpy as np
import scores as scs
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline 

#Loading data
path = '~/Desktop/TFM'
df_train = pd.read_csv(path + '/input/train_users_2.csv')
df_test = pd.read_csv(path + '/input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']

#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
values = df_all.values
#X_train = values[:len(labels)]
X_train = values[:len(labels)-40000]
Xtc = values[len(labels)-40000: len(labels)]
X_test = values[len(labels):]
#mlb = MultiLabelBinarizer(classes=("AU","CA","DE","ES","FR","GB","IT","NDF","NL","PT","US","other"))
#labels.shape = len(labels),1
le = LabelEncoder()
Y_train = le.fit_transform(labels)
Ytc = Y_train[len(labels)-40000:]
Y_train = Y_train[:len(labels)-40000] 
X_test = values[len(labels):]

#Classifier Random Forest
clf = RandomForestClassifier(n_estimators=25, max_depth=6)
clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
y_predictions_test=clf.predict_proba(X_test)

predictionsRF = clf.predict_proba(Xtc)

scoreRF = scs.score_ndcg5(predictionsRF, Ytc, 5)
print scoreRF

#Classifiers Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(X_train, Y_train)
predictionsG = gnb.predict_proba(Xtc)
scoreG = scs.score_ndcg5(predictionsG, Ytc, 5)
print scoreG

#mnb = MultinomialNB()
#mnb = mnb.fit(X_train, Y_train)
#predictionsM = mnb.predict_proba(Xtc)
#scoreM = scs.score_ndcg5(predictionsM, Ytc, 5)
#print scoreM

bnb = BernoulliNB()
bnb = bnb.fit(X_train, Y_train)
predictionsB = bnb.predict_proba(Xtc)
scoreB = scs.score_ndcg5(predictionsB, Ytc, 5)
print scoreB