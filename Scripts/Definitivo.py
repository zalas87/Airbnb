# -*- coding: utf-8 -*-
"""
Created on Sat May 14 16:28:02 2016

@author: family
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 07 13:48:07 2016

@author: gonzalo
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
import scores as scs
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, grid_search

np.random.seed(0)

#(Computing the season for the two dates)
Y = 2000
seasons = [(0, (date(Y,  1,  1),  date(Y,  3, 20))),  #'winter'
           (1, (date(Y,  3, 21),  date(Y,  6, 20))),  #'spring'
           (2, (date(Y,  6, 21),  date(Y,  9, 22))),  #'summer'
           (3, (date(Y,  9, 23),  date(Y, 12, 20))),  #'autumn'
           (0, (date(Y, 12, 21),  date(Y, 12, 31)))]  #'winter'
           
def get_season(dt):
    dt = dt.date()
    dt = dt.replace(year=Y)
    return next(season for season, (start, end) in seasons if start <= dt <= end)

def process_birthday(dataframe):
    list_age = dataframe[(dataframe.age >1900) & (dataframe.age <2010)].age
    
    for i in list_age.index:
        row = dataframe[dataframe.index==i]
        
        #subtract the birthday year from the account create day 
        nage = row.dac_year - row.age
        dataframe.age[dataframe.index==i] = nage
        
    
    return dataframe 
    
def process_sessionsfile(path, df):

    df_session = pd.read_csv(path + "/input/sessions.csv")
    df_session.action = df_session.action.fillna('NAN')
    df_session.action_type = df_session.action_type.fillna('NAN')
    df_session.action_detail = df_session.action_detail.fillna('NAN')
    
    #Action values with low frequency are changed to 'OTHER'
    act_freq = 100  #Threshold for frequency
    act = dict(zip(*np.unique(df_session.action, return_counts=True)))
    df_session.action = df_session.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)
    
    
    groupby_object= df_session.groupby(df_session.user_id, sort=False)
    total_te=groupby_object.sum()
    total_te.columns = ['total_secs_elapsed']
    df = pd.merge(df, total_te, left_on='id', right_on=total_te.index.values, how='left')
    mean_te= groupby_object.mean()
    mean_te.columns = ['mean_secs_elapsed']
    df = pd.merge(df, mean_te, left_on='id', right_on=total_te.index.values, how='left')
    std_te = groupby_object.std()
    std_te.columns = ['std_secs_elapsed']
    df = pd.merge(df, std_te, left_on='id', right_on=total_te.index.values, how='left')
    median_te = groupby_object.median()
    median_te.columns = ['median_secs_elapsed']
    df = pd.merge(df, median_te, left_on='id', right_on=total_te.index.values, how='left')

#    for action in df_session.action.value_counts().index:
#        groupby_object = df_session.groupby(df_session.user_id[df_session.action==action], sort=False)
#        total_sec = groupby_object.sum()
#        total_sec.columns = ['secs_' + action]
#      
#        df = pd.merge(df, total_sec, left_on='id', right_on=total_sec.index.values, how='left')
#
#    for action in df_session.action_detail.value_counts().index:
#        groupby_object = df_session.groupby(df_session.user_id[df_session.action_detail==action], sort=False)
#        total_sec = groupby_object.sum()
#        total_sec.columns = ['secs_' + action]
#      
#        df = pd.merge(df, total_sec, left_on='id', right_on=total_sec.index.values, how='left')
#
#    for action in df_session.action_type.value_counts().index:
#        groupby_object = df_session.groupby(df_session.user_id[df_session.action_type==action], sort=False)
#        total_sec = groupby_object.sum()
#        total_sec.columns = ['secs_' + action]
#      
#        df = pd.merge(df, total_sec, left_on='id', right_on=total_sec.index.values, how='left')
#   
    for action in df_session.action.value_counts().index:
        groupby_object = df_session.groupby(df_session.user_id[df_session.action==action], sort=False)
        total_itec = pd.DataFrame(groupby_object.size(), columns=['iter_' + action])
        df = pd.merge(df, total_itec, left_on='id', right_on=total_itec.index.values, how='left')

    for action in df_session.action_detail.value_counts().index:
        groupby_object = df_session.groupby(df_session.user_id[df_session.action_detail==action], sort=False)
        total_itec = pd.DataFrame(groupby_object.size(), columns=['iter_' + action])
        df = pd.merge(df, total_itec, left_on='id', right_on=total_itec.index.values, how='left')

    for action in df_session.action_type.value_counts().index:
        groupby_object = df_session.groupby(df_session.user_id[df_session.action_type==action], sort=False)
        total_itec = pd.DataFrame(groupby_object.size(), columns=['iter_' + action])
        df = pd.merge(df, total_itec, left_on='id', right_on=total_itec.index.values, how='left')
        
    
    return df    

#Loading data
path = '~/Desktop/TFM'

#train data
df_train = pd.read_csv(path + '/input/train_users_2.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
piv_train = df_train.shape[0]

#test data
df_test = pd.read_csv(path + '/input/test_users.csv')
id_test = df_test['id']

#####Feature engineering#######
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['date_first_booking'], axis=1)

##################Train and test data#######################
#date_account_created
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
dac_dates = [datetime(x[0],x[1],x[2]) for x in dac]

#timestamp_first_active
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
tfa_dates = [datetime(x[0],x[1],x[2]) for x in tfa]

#df_all = process_date_columns(df_all,['date_account_created','timestamp_first_active'] )
df_all['dac_tfa_secs'] = np.array([np.log(1+abs((dac_dates[i]-tfa_dates[i]).total_seconds())) for i in range(len(dac_dates))])
df_all['sig_dac_tfa'] = np.array([np.sign((dac_dates[i]-tfa_dates[i]).total_seconds()) for i in range(len(dac_dates))])
df_all = df_all.drop(['date_account_created','timestamp_first_active'], axis=1)
                    
df_all['season_dac'] = np.array([get_season(dt) for dt in dac_dates])
df_all['season_tfa'] = np.array([get_season(dt) for dt in tfa_dates])

#Age
df_all = process_birthday(df_all)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

##################Session data#####################
df_all = process_sessionsfile(path, df_all)

#Filling nan
df_all = df_all.fillna(-1)

#drop id
df_all = df_all.drop('id', axis=1)
    
    
#Splitting train and test
vals = df_all.values
X_train = vals[:len(labels)-62096]
Xvc = vals[len(labels)-62096: len(labels)]
le = LabelEncoder()
Y_train = le.fit_transform(labels)
Yvc = Y_train[len(labels)-62096:]
y_train = Y_train[:len(labels)-62096] 
X_test = vals[len(labels):]

                                        

#X = vals[:piv_train]
#le = LabelEncoder()
#y = le.fit_transform(labels)   
#X_test = vals[piv_train:]

#Classifier
#clfs = {'LR'  : LogisticRegression(random_state=np.random.seed(0)), 
#        'SVM' : SVC(probability=True, random_state=np.random.seed(0)), 
#        'RF'  : RandomForestClassifier(n_estimators=100, n_jobs=-1, 
#                                       random_state=np.random.seed(0)), 
#        'GBM' : GradientBoostingClassifier(n_estimators=50, 
#                                           random_state=np.random.seed(0)), 
#        'ETC' : ExtraTreesClassifier(n_estimators=100, n_jobs=-1, 
#                                     random_state=np.random.seed(0)),
#        'KNN' : KNeighborsClassifier(n_neighbors=30)}
#for nm, clf in clfs.items():
     #First run. Training on (X_train, y_train) and predicting on X_valid.
#     clf.fit(X_train, y_train)
#     predictions = clf.predict_proba(Xvc)
#     score = scs.score_ndcg5(predictions, Yvc, 5)
#     print np.mean(score)
        
#print '\n'
#Taking the 5 classes with highest probabilities
#ids = []  #list of ids
#cts = []  #list of countries
#for i in range(len(id_test)):
#    idx = id_test[i]
#    ids += [idx] * 5
#    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
#sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
#sub.to_csv('sub.csv',index=False)

#parameters = {'n_estimators':[100,200],'criterion':['gini','entropy']}
#svr = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=np.random.seed(0)) 
#clf = grid_search.GridSearchCV(svr, parameters)#clf = GaussianNB()
#clf = LogisticRegression(penalty='l2',C =0.81, random_state=np.random.seed(0), intercept_scaling=1.1,n_jobs=-1)
#clf = KNeighborsClassifier(n_jobs=-1, algorithm = 'ball_tree', metric='euclidean' , n_neighbors=500)
#clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1,max_features=24, max_depth=None, min_samples_split=1, min_samples_leaf=4, random_state=np.random.seed(0))
#clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_features=24, max_depth=None, min_samples_split=1, min_samples_leaf=4, random_state=np.random.seed(0)) 
#clf = GradientBoostingClassifier(n_estimators=50, max_features=24, max_depth=6, min_samples_split=1, min_samples_leaf=4, random_state=np.random.seed(0))
clf= XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5,  seed=0) 
clf = clf.fit(X_train, y_train)
#print clf.best_params_

predictionsGBC = clf.predict_proba(Xvc)
scoreGBC = scs.score_ndcg5(predictionsGBC, Yvc, 5)
print np.mean(scoreGBC)


#Taking the 5 classes with highest probabilities
#ids = []  #list of ids
#cts = []  #list of countries
#for i in range(len(id_test)):
#    idx = id_test[i]
#    ids += [idx] * 5
#    cts += le.inverse_transform(np.argsort(predictionsGBC[i])[::-1])[:5].tolist()
#
##Generate submission
#sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
#sub.to_csv('subKNN.csv',index=False)