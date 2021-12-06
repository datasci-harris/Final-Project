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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


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
# Linear Regression
####

X = df_democrat.iloc[:, 16].values.reshape(-1, 1) #percentage index starting from 0
Y = df_democrat.iloc[:, 19].values.reshape(-1, 1) #yes approval for biden
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

model =sm.OLS(Y, X).fit()
print_model = model.summary()
print(print_model)     #displaying OLS model summary p values for democrats

####  Multi-Linear Regression

X2 = df_democrat[['pct','sample_size_president','fte_grade','sample_size_governor']]
Y2 = df_democrat['no']

regr = linear_model.LinearRegression()
regr.fit(X2, Y2)

model2 =sm.OLS(Y2, X2).fit()
print_model2 = model2.summary()
print(print_model2)   #insignificant pct result correlation


####
# Machine Learning Supervised:Classfication
####

#split exo X and endo Y into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state= 66)
#linear regression object 
regres = linear_model.LinearRegression()

#train the model
regres.fit(X_train, Y_train)

#score from training
print(regres.score(X_train, Y_train))

#score from testing
print(regres.score(X_test,Y_test))

#trying some GradientBoosteringRegressor Model for better score

regres2 = GradientBoostingRegressor(random_state=66).fit(X_train, Y_train)

#score from training
print(regres2.score(X_train, Y_train))

#score from testing
print(regres2.score(X_test,Y_test))

#trying RandomForestRegressor Model for better score too
regres3 = RandomForestRegressor(random_state=66).fit(X_train, Y_train)

#score from training
print(regres3.score(X_train, Y_train))

#score from testing
print(regres3.score(X_test,Y_test))

###
### RandomForestRegresor test yields the highest Determination of Coefficent
### As known as the R-square. We look at the test score b/c its unbiased
###
#traina and plot
plt.style.use('fivethirtyeight')  #since data source from 538, use its style

## plotting residual errors in training data
plt.scatter(regres.predict(X_train), regres.predict(X_train) - Y_train,
            color = "green", s = 8, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(regres.predict(X_test), regres.predict(X_test) - Y_test,
            color = "blue", s = 8, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 60, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'best')
 
## plot title
plt.title("Residual Errors for Democrats Governor Poll vs Biden Approval Rate")
plt.show()
plt.savefig('machine_learning_resid.png')


####
# Plotting
####
fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.plot(X, Y_pred, color='red')
fig.show()
fig.savefig("regression.png")
fig.clear()

# OR if revealing in Seaborn
sns.lmplot(x='pct',y='yes',data=df_democrat,fit_reg=True) #democrat
sns.lmplot(x='pct',y='yes',data=df_republican,fit_reg=True) #republican

g =sns.lmplot(x="pct", y="yes", col="candidate_party", hue="candidate_party",
               data=df, col_wrap=2, height=5)
g.set(xlim=(0,70),ylim =(0,None),ylabel='Bidens Approval Rate',xlabel = 'Governors Polling Rate')
g.fig.suptitle('Graph')
g.fig.show()
g.fig.savefig('graphs of facegrid.png')
g.fig.clear()


types = df.reset_index()['candidate_party'].unique() # to see how many slices

explode_method=(0, 0.1, 0.1,0,0,0,0,0,0.1,0.1)
explode_party= (0.01,0,0)

####
#Also Plotting the composition of poll types with Generalized function
####

def piechart(df, element, explode):

    plot = df[element].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                       explode = explode,shadow = True, startangle =90,title ='Pie Chart', labeldistance = 1.05)
    return plot

methodology_pie = piechart(df, 'methodology', explode_method) #yielding pie chart

parties_pie = piechart(df, 'candidate_party', explode_party)

####



