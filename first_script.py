# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:26:08 2015

@author: Gonzalo
"""
import pandas as pd
import numpy as np

train_data = pd.read_csv("~/Desktop/TFM/input/train_users_2.csv")
test_data = pd.read_csv("~/Desktop/TFM/input/test_users.csv")

labels = train_data['country_destination'].values
ids_test = test_data['id']

train_data = train_data.drop('country_destination',axis=1)
all_data = pd.concat((train_data,test_data),axis=0, ignore_index=True)
all_data = all_data.drop(['id','date_first_booking'], axis=1)

categorical = ['gender',
               'signup_method',
               'language',
               'affiliate_channel',
               'affiliate_provider',
               'first_affiliate_tracked',
               'signup_app',
               'first_device_type',
               'first_browser',
               'country_destination'
                ]

#Preprocessing the dates
def dateParse(date,c):
    
    try:
        if type(date) == str:
            spl = date.split('-')
        else:
            string = str(date)
            spl = [string[0:4],string[4:6],string[6:8]]
        
        if c == "Y":
            return spl[0]
        elif c == "m":            
            return spl[1]
        else:
            return spl[2]
    except:
        return 0
 
#date_account_created              
all_data['dac_year'] = all_data.date_account_created.apply(dateParse, args=('Y'))  
all_data['dac_month'] = all_data.date_account_created.apply(dateParse, args=('m'))  
all_data['dac_day'] = all_data.date_account_created.apply(dateParse, args=('d'))  

#timestamp_first_active
all_data['tfa_year'] = all_data.timestamp_first_active.apply(dateParse, args=('Y'))
all_data['tfa_month'] = all_data.timestamp_first_active.apply(dateParse,args=('m'))
all_data['tfa_day'] = all_data.timestamp_first_active.apply(dateParse, args=('d'))

all_data = all_data.drop(['date_account_created','timestamp_first_active'], axis=1)
