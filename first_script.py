# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:26:08 2015

@author: Gonzalo
"""
import pandas as pd
import numpy as np
import scores as scs
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline 

def parse_date(date,c=0):
    import time
    
    if type(date) == str:
        spl = date.split('-')
    elif type(date) == np.int64:
        string = str(date)
        spl = [string[0:4],string[4:6],string[6:8]]
    else: 
        return np.nan
        
    if c == "Y":
        return int(spl[0])
    else: 
        date = spl[2] + ' ' + spl[1] + ' ' + spl[0]
        return time.strptime(date, "%d %m %Y").tm_yday

def process_date_columns(dataframe, columns):  
    
    for column in columns:
        fc = list(map(lambda x:x[0],column.split('_')))
        string = ''.join(fc)
             
        dataframe[string + '_year'] = dataframe[column].apply(parse_date, args=('Y'))
        dataframe[string + '_day'] = dataframe[column].apply(parse_date) 
        
    dataframe = dataframe.drop(columns, axis=1)
    return dataframe
    

def add_time_difference(dataframe, st1, st2):
    
    fc = list(map(lambda x:x[0],st1.split('_')))
    string1 = ''.join(fc)    
    fc = list(map(lambda x:x[0],st2.split('_')))
    string2 = ''.join(fc)
    
    string = string1 +'-'+ string2 + '_sec'
    
    t1 = pd.to_datetime(dataframe[st1])
    t2 = pd.to_datetime((dataframe[st2] // 1000000), format='%Y%m%d')
    
    dt = t1-t2 #result in timedelta64 (ns)
    #convert to float64 (s)
    dt = dt.astype(np.int64)/10**9
    
    dataframe[string] = dt
    dataframe = dataframe.drop([st1,st2], axis=1)
    
    return dataframe
    

def add_attrib_totalTimeElapsed(path, dataframe):

    session_data = pd.read_csv(path + '/input/sessions.csv')
    #session_data['secs_elapsed'].fillna(0, inplace=True)
    #sum up elapsed time for each user
    total_te= session_data.groupby(session_data.user_id, sort=False).sum()
    #merge dataframes
    dataframe = pd.merge(dataframe, total_te, left_on='id', right_on=total_te.index.values, how='left')
    
    return dataframe
    
def one_hot_encoder(dataframe, features):
    for feature in features:
         dataframe_dummy = pd.get_dummies(dataframe[feature], prefix=feature)
         dataframe = dataframe.drop([feature], axis=1)
         dataframe = pd.concat((dataframe, dataframe_dummy), axis=1)   

    return dataframe

def visualization(all_data, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="ticks", color_codes=True)

    train_users = pd.concat((all_data[:len(labels)],labels),axis=1)

    #Plot date_first_booking
    #year
    plt.figure(figsize=(16,10))
    sns.countplot(x=train_users.dfb_year,hue=labels, data=train_users,order=[2010,2011,2012,2013,2014,2015], palette="RdBu")
    sns.despine()

    #day
    # cut day values into ranges 
    day_range = pd.cut(train_users["dfb_day"], [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    plt.figure(figsize=(16,10))
    sns.countplot(x=day_range,hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #scatter
    sns.pairplot(train_users, hue="country_destination", markers='o',vars=["dfb_year", "dfb_day"], palette ="RdBu",size=5 )
    
    #Plot gender
    plt.figure(figsize=(16,10))
    sns.countplot(x=train_users['gender'],hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #Plot Signup_method
    plt.figure(figsize=(16,10))
    sns.countplot(x=train_users['signup_method'],hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #Plot Signup_flow
    plt.figure(figsize=(16,10))
    sns.countplot(x=train_users['signup_flow'],hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #Plot Language 
    plt.figure(figsize=(16,10))
    sns.countplot(x=train_users['language'],hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #Plot secs_elapsed --> Don't work
    #plt.figure(figsize=(16,10))
    #sns.countplot(x=train_users['secs_elapsed'],hue=train_users.country_destination, data=train_users, palette="RdBu")
    #sns.despine()

    #Plot Age
    # cut age values into ranges 
    age_range = pd.cut(train_users["age"], [0, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104,2020])

    plt.figure(figsize=(16,10))
    sns.countplot(x=age_range,hue=train_users.country_destination, data=train_users, palette="RdBu")
    sns.despine()

    #Plot dac-tfa
    sns.pairplot(train_users, hue="country_destination", markers='o',vars=["dac-tfa_sec"], palette ="RdBu",size=5 )
    #scatter-years_days_dac-tfa
    sns.pairplot(train_users, hue="country_destination", markers='o',vars=["dfb_year", "dfb_day","dac-tfa_sec"], palette ="RdBu",size=5 )

    return 0

    
    
 #Loading data
path = '~/Desktop/TFM'   
train_data = pd.read_csv(path + '/input/train_users_2.csv')
test_data = pd.read_csv(path + '/input/test_users.csv')
labels = train_data['country_destination'].values
id_test = test_data['id']

train_data = train_data.drop('country_destination',axis=1)
all_data = pd.concat((train_data,test_data),axis=0, ignore_index=True)
#all_data = all_data.drop(['id','date_first_booking'], axis=1)

#Preprocessing the dates
#add the total time elapsed (s) for each user
all_data  = add_attrib_totalTimeElapsed(path, all_data)

#Parse the column date first booking
all_data = process_date_columns(all_data, ['date_first_booking'])

#Sustract dates (date_account_created - timestamp_first_active)
all_data = add_time_difference(all_data, 'date_account_created', 'timestamp_first_active')

#drop some attributes
all_data = all_data.drop(['affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser'], axis=1)

#drop id
all_data = all_data.drop('id', axis=1)
    
#Visualization
#visualization(all_data, labels)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language']
all_data = one_hot_encoder(all_data, ohe_feats)
  
#replace NaN values -> skip error Scikit-learn  
all_data = all_data.fillna(-1)

#Splitting train and test
values = all_data.values
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
