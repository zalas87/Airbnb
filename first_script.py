# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:26:08 2015

@author: Gonzalo
"""
import pandas as pd
import numpy as np

train_data = pd.read_csv("~/Desktop/TFM/input/train_users_2.csv")
test_data = pd.read_csv("~/Desktop/TFM/input/test_users.csv")

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
    

def add_attrib_totalTimeElapsed(dataframe):

    session_data = pd.read_csv("~/Desktop/TFM/input/sessions.csv")
    #sum up elapsed time for each user
    total_te= session_data.groupby(session_data.user_id, sort=False).sum()
    #merge dataframes
    dataframe = pd.merge(dataframe, total_te, left_on='id', right_on=total_te.index.values, how='left')
    
    return dataframe


#Preprocessing the dates

labels = train_data['country_destination'].values

train_data = train_data.drop('country_destination',axis=1)
all_data = pd.concat((train_data,test_data),axis=0, ignore_index=True)
#all_data = all_data.drop(['id','date_first_booking'], axis=1)

#add the total time elapsed (s) for each user
all_data  = add_attrib_totalTimeElapsed(all_data)

#Parse the column date first booking
all_data = process_date_columns(all_data, ['date_first_booking'])

#Sustract dates (date_account_created - timestamp_first_active)
all_data = add_time_difference(all_data, 'date_account_created', 'timestamp_first_active')

#drop some attributes
all_data = all_data.drop(['affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser'], axis=1)

X_train = all_data[:len(labels)].values
y_train = labels
X_test = all_data[len(labels):].values