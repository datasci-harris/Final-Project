# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:27:46 2021

@author: xiaor
"""

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
cwd = os.getcwd()
path =r'C:\Users\xiaor\Documents\GitHub\Final-Project'

fname1 = 'governor_polls.csv'
fname2 = 'president_approval_polls.csv'

governor = pd.read_csv(os.path.join(path, fname1))
president = pd.read_csv(os.path.join(path,fname2))

col1 = governor.columns #show numbers of columns for governor
col2 = president.columns #show columns of president


####
# Data Cleaning
####

#since only interested in swing state supporting rate, begin data cleaning
# by dropping other states
state_list = ['Virginia','New Jersey']  #wanted Swing state
df_gov =governor[governor['state'].isin(state_list)] 
#df_gov represents the swing_state governor support rate that we were interested in

gov_droplist = ['question_id','poll_id','cycle','sponsor_ids','sponsors','seat_number','seat_name','start_date','end_date','election_date','sponsor_candidate','internal']
df_gov =df_gov.drop(columns = gov_droplist)

# we are interested in a couple more concise framework, so we will drop more
df_gov = df_gov.drop(df_gov.columns[16:21],axis = 1)

# Now we will try to merge the remaining data with president
# Data cleaning for president as well
select_list = ['politician_id','politician','pollster_id','pollster','display_name','pollster_rating_id','pollster_rating_name','fte_grade','sample_size','methodology','yes','no','alternate_answers']
df_psd = president[select_list]

df = pd.merge(df_gov, df_psd, how='right', on = ['pollster_id','pollster','display_name','pollster_rating_id','pollster_rating_name','fte_grade','methodology'])
df = df.drop(df.columns[12:15], axis = 1)  #drop three unnessary columns
df = df.drop(['politician_id','candidate_id'], axis =1) #some more

# renaming size and candidate columns to clarify between presidents and governor candidates
df =df.rename(columns = {'sample_size_x':'sample_size_governor','sample_size_y':'sample_size_president','politician':'president'})
# replacing column fte_grading with numerical value for better analyze
fte_dict = {'A+':4.3,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,'C+':2.3,'C':2.0,'C-':1.7,'D+':1.3,'D':1.0,'A/B':3.5,'B/C':2.5,'C/D':1.5,}
df = df.replace({'fte_grade':fte_dict}) #five thirty eight column replaced with numerical, referenced from UChicago GPA agenda

df_democrat = df[df['candidate_party'] == 'DEM']
df_republican = df[df['candidate_party'] == 'REP']
####
# Regression
####

X = df_democrat.iloc[:, 16].values.reshape(-1, 1) #percentage index starting from 0
Y = df_democrat.iloc[:, 19].values.reshape(-1, 1) #yes approval for biden
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

####
# Plotting
####
plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

# OR if revealing in Seaborn
sns.lmplot(x='pct',y='yes',data=df_democrat,fit_reg=True) #democrat
sns.lmplot(x='pct',y='no',data=df_republican,fit_reg=True) #republican
