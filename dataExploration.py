# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 11:27:22 2015

@author: family
"""

#Exploración de los datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("~/Desktop/TFM/input/train_users_2.csv")
test_data = pd.read_csv("~/Desktop/TFM/input/test_users.csv")

train_data = train_data.drop(['id','date_first_booking'], axis=1)
test_data = test_data.drop(['id','date_first_booking'], axis=1)

train_users = train_data.shape[0]
test_users = test_data.shape[0]
print "Users in the training set: ", train_users
print "\nUsers in the training set: ", test_users
print "\nTotal registros:", (train_users + test_users)

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
                
#                
train_data['date_account_created'] = pd.to_datetime(train_data['date_account_created'])
test_data['date_account_created'] = pd.to_datetime(test_data['date_account_created'])

train_data['timestamp_first_active'] = pd.to_datetime((train_data.timestamp_first_active // 1000000), format='%Y%m%d')
test_data['timestamp_first_active'] = pd.to_datetime((test_data.timestamp_first_active // 1000000), format='%Y%m%d')


for feature in categorical:
    #Plot categorical features
    plt.figure() 
    if feature != 'country_destination':
        #Training set
        percent = (train_data[feature].value_counts()/train_users)*100 
        percent.plot(kind='bar', color='#FD5C64', rot=0)
        plt.xticks(rotation = 'vertical')
        plt.title('Training set')
        plt.xlabel(feature)
        plt.ylabel('Percentage')    
        sns.despine()
        
        plt.figure() 
        #Test set
        percent = (test_data[feature].value_counts()/test_users)*100 
        percent.plot(kind='bar', color='#FD5C64', rot=0)
        plt.xticks(rotation = 'vertical')
        plt.title('Test set')
        plt.xlabel(feature)
        plt.ylabel('Percentage')     
        sns.despine()
    else:
       percent = (train_data[feature].value_counts()/train_users)*100 
       percent.plot(kind='bar', color='#FD5C64', rot=0)
       plt.xticks(rotation = 'vertical')
       plt.xlabel(feature)
       plt.ylabel('Percentage')    
       sns.despine()
       
       
#Plot dates   
plt.figure() 
sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')       
test_data.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#63EA55')
plt.xlabel('date account create')

plt.figure() 
sns.set_style("whitegrid", {'axes.edgecolor': '0'})
sns.set_context("poster", font_scale=1.1)
train_data.timestamp_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
test_data.timestamp_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#63EA55')
plt.xlabel('timestamp first active')

#Plot signud_flow
plt.figure() 
percent = (train_data['signup_flow'].value_counts(sort=False)/train_users)*100 
percent.plot(kind='bar', color='#FD5C64', rot=0)
plt.title('Training set')
plt.xlabel('Signup flow')
plt.ylabel('Percentage')    
sns.despine()

plt.figure() 
percent = (test_data['signup_flow'].value_counts(sort=False)/train_users)*100 
percent.plot(kind='bar', color='#FD5C64', rot=0)
plt.title('Test set')
plt.xlabel('Signup flow')
plt.ylabel('Percentage')    
sns.despine()

#Plot Age

av0 = train_data[(train_data.age <= 100) & (train_data.age >= 15)].age
plt.figure() 
sns.distplot(av0, color='#FD5C64')
plt.title('train_set')
plt.xlabel('Age')
sns.despine()

av1 = test_data[(test_data.age <= 100) & (test_data.age >= 15)].age
plt.figure() 
sns.distplot(av1, color='#FD5C64')
plt.title('test_set')
plt.xlabel('Age')
sns.despine()

op0 = train_data[train_data.age < 15]
plt.figure() 
percent = (op0['country_destination'].value_counts()/op0.shape[0])*100 
percent.plot(kind='bar', color='#FD5C64', rot=0)
plt.title('train_set: Age < 15')
plt.xlabel('country_destination')
plt.ylabel('Percentage')    
sns.despine()

op1 = train_data[train_data.age > 100]
plt.figure() 
percent = (op1['country_destination'].value_counts()/op1.shape[0])*100 
percent.plot(kind='bar', color='#FD5C64', rot=0)
plt.title('train_set: Age >100')
plt.xlabel('country_destination')
plt.ylabel('Percentage')    
sns.despine()