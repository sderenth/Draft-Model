# Started: 12.12.17
# Finished: 12.12.17
# Last Updated:
#
# Draft Model v2
#
# Revisions:
#   1. Added Hieght to data
#   2. Added Personal Fouls per Minute to data
#   3. Experimented with minimum minutes limits.
#   4. Decided on -10.0 BPM for players that weren't drafted or didn't play at least 4 years in NBA.
# 
# by Sean Derenthal

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# set working directory
os.chdir('C:\Users\Sean\BasketballData\RealGM Data from Will')

# read data into DataFrames
df = pd.DataFrame(pd.read_csv('RealGMDataClean.csv'))



### DATA PREP ###

# Drop 'Unnamed: 0' (which was the index from cleaning file's dataframe) and 'Player ID'.
df = df.drop(['Unnamed: 0', 'Player ID'], axis=1)

# Fill NAs of 'FGA' with 0
df['FGA'] = df['FGA'].fillna(0)

# Fill NAs of 'BPM' with -10
df['BPM'] = df['BPM'].fillna(-10.0)


## Deal with following columns that have missing values: 'ORB%', 'height_inches', 'pf_min', 'adjORB%'.

# 'pf_min': MIN value was 0.0. So we'll just remove rows with MIN == 0.0.
df = df[df['MIN'] != 0]

# 'ORB%' was originally scrapped improperly. Remove both ORB% columns.
df.drop(['ORB%', 'adjORB%'], axis=1, inplace=True)

# 'height_inches' is missing 6 values, so we'll just add them manually.
df.at[1255, 'height_inches'] = 73
df.at[2631, 'height_inches'] = 80
df.at[3818, 'height_inches'] = 73
df.at[6155, 'height_inches'] = 76
df.at[10501, 'height_inches'] = 79
df.at[11512, 'height_inches'] = 73


### MODEL BUILDING ###

# Assign target variable and predictors.
# y = df['BPM']
# x = df.drop('BPM', axis=1)

# Starting with v1, but with all data, including 'height_inches' and 'pf_min'
# model = RandomForestRegressor(500, oob_score=True, n_jobs=-1, random_state=1, min_samples_leaf=5)
# model.fit(x, y)
# print 'Model 2, R^2: ', model.oob_score_
# print ''

# Variable Strength
# Graph showing the relative importance of each variable on the final prediction of the model.
# var_str = pd.Series(model.feature_importances_, index=x.columns)
# var_str.sort_values(inplace=True)
# var_str.plot(kind='barh', grid=True)
# plt.show()


# Optimize attributes used.
# model = RandomForestRegressor(500, oob_score=True, n_jobs=-1, random_state=1, min_samples_leaf=5)
# model.fit(x, y)
# var_str = pd.Series(model.feature_importances_, index=x.columns)
# 
# x_attributes = var_str.sort_values(ascending=False).index.tolist()
# 
# r2_scores = []
# attr_num_removed = range(-34, -42, -1) # To narrow to this range, previously used 'range(-1, -53, -5)'.
# for num in attr_num_removed:
#     x = df[x_attributes[:num]]    
#     model = RandomForestRegressor(n_estimators = 500,
#                                   oob_score=True,
#                                   n_jobs=-1,
#                                   random_state=1,
#                                   min_samples_leaf=5)
#     model.fit(x, y)
#     print num, 'attributes removed.'
#     r2 = model.oob_score_
#     print 'R^2: ', r2
#     r2_scores.append(r2)
#     print ''
#     
# pd.Series(r2_scores, attr_num_removed).plot()
# plt.show()

# Conclusion: removing the 40 "least important" attributes gives the best results.


# Results of above optimization:
# x= df[['adjMIN',
#  'adjPER',
#  'GP',
#  'age',
#  'DRtg',
#  'adjeFG%',
#  'adjDRtg',
#  'adjORtg',
#  'PER',
#  'BLK',
#  'GS',
#  'FGM',
#  'DEF',
#  'AST',
#  'PPR']]
# 
# model = RandomForestRegressor(500, oob_score=True, n_jobs=-1, random_state=1, min_samples_leaf=5)
# model.fit(x, y)
# print 'Model 2, R^2: ', model.oob_score_
# print ''
# 
# # Variable Strength
# # Graph showing the relative importance of each variable on the final prediction of the model.
# var_str = pd.Series(model.feature_importances_, index=x.columns)
# var_str.sort_values(inplace=True)
# var_str.plot(kind='barh', grid=True)
# plt.show()


### Revision: let's try cutting out the players that played less than a certian number of average minutes per year.
# important_attr = ['adjMIN','adjPER','GP','age','DRtg','adjeFG%','adjDRtg','adjORtg','PER','BLK','GS','FGM','DEF','AST','PPR']
# minutes_minimums = range(50, 1101, 50)
# r2_scores = []
# for minutes_minimum in minutes_minimums:
#     new_df = df[df['adjMIN'] >= minutes_minimum]
#     y = new_df['BPM']    
#     x = new_df[important_attr]   
#     model = RandomForestRegressor(n_estimators = 500,
#                                   oob_score=True,
#                                   n_jobs=-1,
#                                   random_state=1,
#                                   min_samples_leaf=5)
#     model.fit(x, y)
#     print minutes_minimum, 'minimum minutes'
#     r2 = model.oob_score_
#     print 'R^2: ', r2
#     r2_scores.append(r2)
#     print ''
#     
# pd.Series(r2_scores, minutes_minimums).plot()
# plt.show()

# Conclusion: Took me a while to understand why but as the minutes limit increased, the model got better. 
#   This is to be expected. As you look at guys who have played more and more minutes, they are easier to predict going forward. 
#   Less variation the bigger the sample size.
#
#   Maybe I can predict each player based on the model trained on a matching adjMIN? 
#   My instinct tells me that this is not sound, but I can't say exactly why. I'll have to think about this.


## Revision: The filler BPM of -6.0 might be too high. Let's try other values and see how that helps the model.
# important_attr = ['adjMIN','adjPER','GP','age','DRtg','adjeFG%','adjDRtg','adjORtg','PER','BLK','GS','FGM','DEF','AST','PPR']
# x = df[important_attr]
# 
# r2_scores = []
# bpms = np.arange(-12.0, -20.0, -1.0)
# 
# for bpm in bpms:
#     y = df['BPM'].fillna(bpm)
#     
#     model = RandomForestRegressor(n_estimators = 500,
#                                   oob_score=True,
#                                   n_jobs=-1,
#                                   random_state=1,
#                                   min_samples_leaf=5)
#     model.fit(x, y)
#     print bpm, 'bpm'
#     r2 = model.oob_score_
#     print 'R^2: ', r2
#     r2_scores.append(r2)
#     print ''
# 
# pd.Series(r2_scores, bpms).plot()
# plt.show()

# Conclusion: -10 BPM is the optimal target variable for players that didn't make the league or didn't last 4 years.
#   Of course, there is probably a better way, than to simply give them all the same value. Although, theoretically, 
#   decision tree models should be able to handle this better because it's like a piece-wise function. Sort of a regression
#   analysis mixed with a classifier.

 


# Final model v 2, after tinkering.
df = df[df['adjMIN'] >= 400]
important_attr = ['adjMIN','adjPER','GP','age','DRtg','adjeFG%','adjDRtg','adjORtg','PER','BLK','GS','FGM','DEF','AST','PPR']
x = df[important_attr]
y = df['BPM']

model = RandomForestRegressor(500, oob_score=True, n_jobs=-1, random_state=1, min_samples_leaf=5)
model.fit(x, y)
print 'Model 2, R^2: ', model.oob_score_
print ''

# Variable Strength
# Graph showing the relative importance of each variable on the final prediction of the model.
var_str = pd.Series(model.feature_importances_, index=x.columns)
var_str.sort_values(inplace=True)
var_str.plot(kind='barh', grid=True)
plt.show()




### Predictions for 2018 ###

# read data into DataFrames
# two018_df = pd.DataFrame(pd.read_csv('2018RealGMDataClean.csv'))
# 
# two018_df = two018_df[two018_df['adjMIN'] >= 400]
# important_attr = ['adjMIN','adjPER','GP','age','DRtg','adjeFG%','adjDRtg','adjORtg','PER','BLK','GS','FGM','DEF','AST','PPR']
# x_2018 = two018_df[important_attr]
# 
# pred_BPM = model.predict(x_2018)
# 
# two018_df['pred_BPM'] = pred_BPM
# 
# two018_df['Player Name'] = two018_df['Player ID'].str.split('/').str[2]
# 
# two018_df.to_csv('2018PredBPM.csv')