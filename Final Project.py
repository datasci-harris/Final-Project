# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:27:46 2021

@author: xiaor
"""

import pandas as pd
import os

cwd = os.getcwd()
fname1 = 'governor_polls.csv'
fname2 = 'president_approval_polls.csv'

governor = pd.read_csv(os.path.join(cwd, fname1))
president = pd.read_csv(os.path.join(cwd,fname2))

col1 = governor.columns #show numbers of columns for governor
col2 = president.columns #show columns of president

#since only interested in swing state supporting rate, begin data cleaning
# by dropping other states
state_list = ['Virginia','New Jersey']  #wanted Swing state
df_gov =governor[governor['state'].isin(state_list)] 
#df_gov represents the swing_state governor support rate that we were interested in

gov_droplist = ['question_id','poll_id','cycle','sponsor_ids','sponsors','seat_number','seat_name','start_date','end_date','election_date','sponsor_candidate','internal']
df_gov =df_gov.drop(columns = gov_droplist)

# we are interested in a couple more concise framework, so we will drop more

df_gov = df_gov.drop(df_gov.columns[16:20],axis = 1)
