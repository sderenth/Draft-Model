# Started: 12.5.17?
# Finished: 12.12.17
# Last Updated:
#
# Clean college data from RealGM, age adjust some variables, and add mean NBA BPM from years 3 and 4.
# 
# by Sean Derenthal


import os
import pandas as pd
import numpy as np

# set working directory
os.chdir('C:\Users\Sean\BasketballData\RealGM Data from Will')

# read data into DataFrames
totals = pd.DataFrame(pd.read_csv('NCAA Player Stats - Totals.csv'))
advanced = pd.DataFrame(pd.read_csv('NCAA Player Stats - Advanced.csv'))
all_player_info = pd.DataFrame(pd.read_csv('RealGM PlayerInfo.csv'))

# drop unimportant attributes.
totals.drop(['Table', 'League', 'Conference', 'Team Name', 'Team Num', 
             'Season', 'Team Polished', 'Class', 'Class w Trans RS', 'Player ID w Class'], axis=1, inplace=True)

advanced.drop(['Player Name','Player Season Num', 'Table', 'League', 'Conference', 'Team Name', 'Team Num', 
               'Season', 'Team Polished', 'Team w Stuff', 'Status', 'Class', 
               'Class w Trans RS', 'Player ID w Class', 'GP', 'GS'], axis=1, inplace=True)

# merge sets
college_player_season_df = pd.merge(totals, advanced, on=['Player ID', '2Y Season'])

player_info = all_player_info[['Player ID', 'Position', 'Height', 'Birthday Code']]

college_player_season_df = pd.merge(college_player_season_df, player_info, how='left', on='Player ID')

# Remove seasons where no games were played.
college_player_season_df = college_player_season_df[college_player_season_df['MIN'] != '-']
college_player_season_df = college_player_season_df[college_player_season_df['USG%'] != '-']

# Change stat columns to numeric types.
str_to_num = ['GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG%', '3PM', '3PA', '3P%', 'FTM',
       'FTA', 'FT%', 'OFF', 'DEF', 'TRB', 'AST', 'STL', 'BLK', 'PF',
       'TOV', 'PTS', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%',
       'TOV%', 'STL%', 'BLK%', 'USG%', 'Total S %', 'PPR', 'PPS',
       'ORtg', 'DRtg', 'PER']

for column in str_to_num:
    college_player_season_df[column] = pd.to_numeric(college_player_season_df[column], errors='coerce')



# Convert string height (ft'inches") to int (inches)
college_player_season_df = college_player_season_df[college_player_season_df['Height'] != '-']
college_player_season_df['height_inches'] = pd.to_numeric(college_player_season_df['Height'].str[0]) * 12 + pd.to_numeric(college_player_season_df['Height'].str.split('-').str[1])

# add age for each season. 'age' is age on Jan 1 of that season.
college_player_season_df['season'] = pd.to_numeric(college_player_season_df['2Y Season'].str[0:2] + college_player_season_df['2Y Season'].str[-2:]) 
college_player_season_df.replace(1900, 2000, inplace=True)

college_player_season_df['Birthday Code'].replace('-', np.nan, inplace=True)

college_player_season_df['age'] = (college_player_season_df.season - pd.to_numeric(college_player_season_df['Birthday Code'].str[:4])) - ((((pd.to_numeric(college_player_season_df['Birthday Code'].str[4:6]) - 1)*30.4375) + pd.to_numeric(college_player_season_df['Birthday Code'].str[6:])) / 365.25)

# Remove players who played in college after the 2012 season.
player_last_season = college_player_season_df.groupby(college_player_season_df['Player ID'])['season'].max()
players_to_delete = player_last_season.isin(range(2013, 2018)).reset_index().rename(columns={'season':'delete_me'})
college_player_season_df = college_player_season_df.merge(players_to_delete, how='left', on='Player ID')
college_player_season_df = college_player_season_df[college_player_season_df['delete_me'] == False]
college_player_season_df.drop('delete_me', axis=1, inplace=True)

# Add pf/min column.
college_player_season_df['pf_min'] = college_player_season_df['PF']/college_player_season_df['MIN']

# To fill in missing ages, give first season the average age of all first year players, and so on.
college_player_season_df['season_num'] = pd.to_numeric(college_player_season_df['Player Season Num'].str.split('_').str[3].str[:-1]).astype(int)

for num in range(1,11):
    num_age_mean = college_player_season_df[college_player_season_df['season_num'] == num]['age'].mean()
    college_player_season_df.loc[college_player_season_df.season_num == num, 'age'] = college_player_season_df.loc[college_player_season_df.season_num == num, 'age'].fillna(num_age_mean)


### Simple regression for age progression adjustments
bins = np.arange(13,32,.25)

age_bins = pd.cut(college_player_season_df['age'], bins)
mean_age_df = college_player_season_df.groupby(age_bins).mean().dropna()


age_groups = mean_age_df.index
x_age_groups = np.asarray([float(interval.split(',')[0][1:])+.125 for interval in age_groups])



# MIN specific adjustments
y_min = np.asarray(mean_age_df['MIN'])
y_min = np.delete(y_min, [1,4])[:-14]

x_min = np.delete(x_age_groups, [1,4])[:-14]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_min = np.polyfit(x_min, y_min, deg=2)
# ax.plot(x_min, fit_min[0] * x_min ** 2 + fit_min[1] * x_min + fit_min[2], color='red')
# ax.scatter(x_min, y_min)
# 
# fig.show()


mean_19_min = fit_min[0] * 19 ** 2 + fit_min[1] * 19 + fit_min[2]
college_player_season_df['adjMIN'] = (college_player_season_df['MIN'] * mean_19_min) / (fit_min[0] * college_player_season_df.age ** 2 + fit_min[1] * college_player_season_df.age + fit_min[2])


# FG% specific adjustments
y_fg_perc = np.asarray(mean_age_df['FG%'])
y_fg_perc = y_fg_perc[6:-16]

x_fg_perc = x_age_groups[6:-16]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_fg_perc = np.polyfit(x_fg_perc, y_fg_perc, deg=1)
# ax.plot(x_fg_perc, fit_fg_perc[0] * x_fg_perc + fit_fg_perc[1], color='red')
# ax.scatter(x_fg_perc, y_fg_perc)
# 
# fig.show()


mean_19_fg_perc = fit_fg_perc[0] * 19 + fit_fg_perc[1]
college_player_season_df['adjFG%'] = (college_player_season_df['FG%'] * mean_19_fg_perc) / (fit_fg_perc[0] * college_player_season_df.age + fit_fg_perc[1])


# 3P% specific adjustments
y_3p_perc = np.asarray(mean_age_df['3P%'])
y_3p_perc = y_3p_perc[5:-15]

x_3p_perc = x_age_groups[5:-15]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_3p_perc = np.polyfit(x_3p_perc, y_3p_perc, deg=2)
# ax.plot(x_3p_perc, fit_3p_perc[0] * x_3p_perc ** 2 + fit_3p_perc[1] * x_3p_perc + fit_3p_perc[2], color='red')
# ax.scatter(x_3p_perc, y_3p_perc)
# 
# fig.show()


mean_19_3p_perc = fit_3p_perc[0] * 19 ** 2 + fit_3p_perc[1] * 19 + fit_3p_perc[2]
college_player_season_df['adj3P%'] = (college_player_season_df['3P%'] * mean_19_3p_perc) / (fit_3p_perc[0] * college_player_season_df.age **2 + fit_3p_perc[1] * college_player_season_df.age + fit_3p_perc[2])



# FT% specific adjustments
y_ft_perc = np.asarray(mean_age_df['FT%'])
y_ft_perc = y_ft_perc[6:-19]

x_ft_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_ft_perc = np.polyfit(x_ft_perc, y_ft_perc, deg=2)
# ax.plot(x_ft_perc, fit_ft_perc[0] * x_ft_perc ** 2 + fit_ft_perc[1] * x_ft_perc + fit_ft_perc[2], color='red')
# ax.scatter(x_ft_perc, y_ft_perc)
# 
# fig.show()


mean_19_ft_perc = fit_ft_perc[0] * 19 ** 2 + fit_ft_perc[1] * 19 + fit_ft_perc[2]
college_player_season_df['adjFT%'] = (college_player_season_df['FT%'] * mean_19_ft_perc) / (fit_ft_perc[0] * college_player_season_df.age **2 + fit_ft_perc[1] * college_player_season_df.age + fit_ft_perc[2])


# TS% specific adjustments
y_ts_perc = np.asarray(mean_age_df['TS%'])
y_ts_perc = y_ts_perc[6:-19]

x_ts_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_ts_perc = np.polyfit(x_ts_perc, y_ts_perc, deg=2)
# ax.plot(x_ts_perc, fit_ts_perc[0] * x_ts_perc ** 2 + fit_ts_perc[1] * x_ts_perc + fit_ts_perc[2], color='red')
# ax.scatter(x_ts_perc, y_ts_perc)
# 
# fig.show()


mean_19_ts_perc = fit_ts_perc[0] * 19 ** 2 + fit_ts_perc[1] * 19 + fit_ts_perc[2]
college_player_season_df['adjTS%'] = (college_player_season_df['TS%'] * mean_19_ts_perc) / (fit_ts_perc[0] * college_player_season_df.age **2 + fit_ts_perc[1] * college_player_season_df.age + fit_ts_perc[2])


# eFG% specific adjustments
y_efg_perc = np.asarray(mean_age_df['eFG%'])
y_efg_perc = y_efg_perc[6:-19]

x_efg_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_efg_perc = np.polyfit(x_efg_perc, y_efg_perc, deg=2)
# ax.plot(x_efg_perc, fit_efg_perc[0] * x_efg_perc ** 2 + fit_efg_perc[1] * x_efg_perc + fit_efg_perc[2], color='red')
# ax.scatter(x_efg_perc, y_efg_perc)
# 
# fig.show()


mean_19_efg_perc = fit_efg_perc[0] * 19 ** 2 + fit_efg_perc[1] * 19 + fit_efg_perc[2]
college_player_season_df['adjeFG%'] = (college_player_season_df['eFG%'] * mean_19_efg_perc) / (fit_efg_perc[0] * college_player_season_df.age **2 + fit_efg_perc[1] * college_player_season_df.age + fit_efg_perc[2])


# ORB% specific adjustments
y_orb_perc = np.asarray(mean_age_df['ORB%'])
y_orb_perc = y_orb_perc[9:-19]

x_orb_perc = x_age_groups[9:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_orb_perc = np.polyfit(x_orb_perc, y_orb_perc, deg=1)
# ax.plot(x_orb_perc, fit_orb_perc[0] * x_orb_perc + fit_orb_perc[1], color='red')
# ax.scatter(x_orb_perc, y_orb_perc)
# 
# fig.show()


mean_19_orb_perc = fit_orb_perc[0] * 19 + fit_orb_perc[1]
college_player_season_df['adjORB%'] = (college_player_season_df['ORB%'] * mean_19_orb_perc) / (fit_orb_perc[0] * college_player_season_df.age + fit_orb_perc[1])


# DRB% specific adjustments
y_drb_perc = np.asarray(mean_age_df['DRB%'])
y_drb_perc = y_drb_perc[6:-19]

x_drb_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_drb_perc = np.polyfit(x_drb_perc, y_drb_perc, deg=1)
# ax.plot(x_drb_perc, fit_drb_perc[0] * x_drb_perc + fit_drb_perc[1], color='red')
# ax.scatter(x_drb_perc, y_drb_perc)
# 
# fig.show()


mean_19_drb_perc = fit_drb_perc[0] * 19 + fit_drb_perc[1]
college_player_season_df['adjDRB%'] = (college_player_season_df['DRB%'] * mean_19_drb_perc) / (fit_drb_perc[0] * college_player_season_df.age + fit_drb_perc[1])


# TRB% specific adjustments
y_trb_perc = np.asarray(mean_age_df['TRB%'])
y_trb_perc = y_trb_perc[6:-19]

x_trb_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_trb_perc = np.polyfit(x_trb_perc, y_trb_perc, deg=1)
# ax.plot(x_trb_perc, fit_trb_perc[0] * x_trb_perc + fit_trb_perc[1], color='red')
# ax.scatter(x_trb_perc, y_trb_perc)
# fig.show()

mean_19_trb_perc = fit_trb_perc[0] * 19 + fit_trb_perc[1]
college_player_season_df['adjTRB%'] = (college_player_season_df['TRB%'] * mean_19_trb_perc) / (fit_trb_perc[0] * college_player_season_df.age + fit_trb_perc[1])


# AST% specific adjustments
y_ast_perc = np.asarray(mean_age_df['AST%'])
y_ast_perc = y_ast_perc[7:-19]

x_ast_perc = x_age_groups[7:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_ast_perc = np.polyfit(x_ast_perc, y_ast_perc, deg=2)
# ax.plot(x_ast_perc, fit_ast_perc[0] * x_ast_perc ** 2 + fit_ast_perc[1] * x_ast_perc + fit_ast_perc[2], color='red')
# ax.scatter(x_ast_perc, y_ast_perc)
# fig.show()

mean_19_ast_perc = fit_ast_perc[0] * 19 ** 2 + fit_ast_perc[1] * 19 + fit_ast_perc[2]
college_player_season_df['adjAST%'] = (college_player_season_df['AST%'] * mean_19_ast_perc) / (fit_ast_perc[0] * college_player_season_df.age ** 2 + fit_ast_perc[1] * college_player_season_df.age + fit_ast_perc[2])


# TOV% specific adjustments
y_tov_perc = np.asarray(mean_age_df['TOV%'])
y_tov_perc = y_tov_perc[7:-22]

x_tov_perc = x_age_groups[7:-22]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_tov_perc = np.polyfit(x_tov_perc, y_tov_perc, deg=1)
# ax.plot(x_tov_perc, fit_tov_perc[0] * x_tov_perc + fit_tov_perc[1], color='red')
# ax.scatter(x_tov_perc, y_tov_perc)
# fig.show()

mean_19_tov_perc = fit_tov_perc[0] * 19 + fit_tov_perc[1]
college_player_season_df['adjTOV%'] = (college_player_season_df['TOV%'] * mean_19_tov_perc) / (fit_tov_perc[0] * college_player_season_df.age + fit_tov_perc[1])


# STL% specific adjustments
y_stl_perc = np.asarray(mean_age_df['STL%'])
y_stl_perc = y_stl_perc[7:-19]

x_stl_perc = x_age_groups[7:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_stl_perc = np.polyfit(x_stl_perc, y_stl_perc, deg=2)
# ax.plot(x_stl_perc, fit_stl_perc[0] * x_stl_perc ** 2 + fit_stl_perc[1] * x_stl_perc + fit_stl_perc[2], color='red')
# ax.scatter(x_stl_perc, y_stl_perc)
# fig.show()

mean_19_stl_perc = fit_stl_perc[0] * 19 ** 2 + fit_stl_perc[1] * 19 + fit_stl_perc[2]
college_player_season_df['adjSTL%'] = (college_player_season_df['STL%'] * mean_19_stl_perc) / (fit_stl_perc[0] * college_player_season_df.age ** 2 + fit_stl_perc[1] * college_player_season_df.age + fit_stl_perc[2])


# BLK% specific adjustments
y_blk_perc = np.asarray(mean_age_df['BLK%'])
y_blk_perc = y_blk_perc[10:-22]

x_blk_perc = x_age_groups[10:-22]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_blk_perc = np.polyfit(x_blk_perc, y_blk_perc, deg=1)
# ax.plot(x_blk_perc, fit_blk_perc[0] * x_blk_perc + fit_blk_perc[1], color='red')
# ax.scatter(x_blk_perc, y_blk_perc)
# fig.show()

mean_19_blk_perc = fit_blk_perc[0] * 19 + fit_blk_perc[1]
college_player_season_df['adjBLK%'] = (college_player_season_df['BLK%'] * mean_19_blk_perc) / (fit_blk_perc[0] * college_player_season_df.age + fit_blk_perc[1])


# USG% specific adjustments
y_usg_perc = np.asarray(mean_age_df['USG%'])
y_usg_perc = y_usg_perc[6:-19]

x_usg_perc = x_age_groups[6:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_usg_perc = np.polyfit(x_usg_perc, y_usg_perc, deg=1)
# ax.plot(x_usg_perc, fit_usg_perc[0] * x_usg_perc + fit_usg_perc[1], color='red')
# ax.scatter(x_usg_perc, y_usg_perc)
# fig.show()

mean_19_usg_perc = fit_usg_perc[0] * 19 + fit_usg_perc[1]
college_player_season_df['adjUSG%'] = (college_player_season_df['USG%'] * mean_19_usg_perc) / (fit_usg_perc[0] * college_player_season_df.age + fit_usg_perc[1])


# ORtg specific adjustments
y_ortg = np.asarray(mean_age_df['ORtg'])
y_ortg = y_ortg[7:-19]

x_ortg = x_age_groups[7:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_ortg = np.polyfit(x_ortg, y_ortg, deg=1)
# ax.plot(x_ortg, fit_ortg[0] * x_ortg + fit_ortg[1], color='red')
# ax.scatter(x_ortg, y_ortg)
# fig.show()

mean_19_ortg = fit_ortg[0] * 19 + fit_ortg[1]
college_player_season_df['adjORtg'] = (college_player_season_df['ORtg'] * mean_19_ortg) / (fit_ortg[0] * college_player_season_df.age + fit_ortg[1])


# DRtg specific adjustments
y_drtg = np.asarray(mean_age_df['DRtg'])
y_drtg = y_drtg[7:-19]

x_drtg = x_age_groups[7:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_drtg = np.polyfit(x_drtg, y_drtg, deg=1)
# ax.plot(x_drtg, fit_drtg[0] * x_drtg + fit_drtg[1], color='red')
# ax.scatter(x_drtg, y_drtg)
# fig.show()

mean_19_drtg = fit_drtg[0] * 19 + fit_drtg[1]
college_player_season_df['adjDRtg'] = (college_player_season_df['DRtg'] * mean_19_drtg) / (fit_drtg[0] * college_player_season_df.age + fit_drtg[1])


# PER specific adjustments
y_per = np.asarray(mean_age_df['PER'])
y_per = y_per[8:-19]

x_per = x_age_groups[8:-19]

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
fit_per = np.polyfit(x_per, y_per, deg=1)
# ax.plot(x_per, fit_per[0] * x_per + fit_per[1], color='red')
# ax.scatter(x_per, y_per)
# fig.show()

mean_19_per = fit_per[0] * 19 + fit_per[1]
college_player_season_df['adjPER'] = (college_player_season_df['PER'] * mean_19_per) / (fit_per[0] * college_player_season_df.age + fit_per[1])



# Combine all college years
mean_college_season = college_player_season_df.groupby(college_player_season_df['Player ID']).mean()
mean_college_season.drop(['season', 'season_num'], axis=1, inplace=True)
mean_college_season.reset_index(inplace=True)


### Get NBA BPM from NBA data set ###
# read data into DataFrames
nba_df = pd.DataFrame(pd.read_csv('Seasons_WS_BPM.csv'))

# Keep years 2003-2017
nba_df = nba_df[nba_df['Year'].isin(range(2004, 2018))]

# Remove partial seasons, and keep season total. Explanation: if a player played for two teams during a single season, there are three
# rows for that player. The first is the total season data, and the second two are data from each team.
nba_df['delete_me'] = np.where(nba_df['Player'] == nba_df['Player'].shift(1), True, False)
nba_df = nba_df[nba_df['delete_me'] == False]
nba_df.drop('delete_me', axis=1, inplace=True)

# Get list of players' BPMs, and make series with mean of 3rd and 4th BPM.
nba_bpm_lists = nba_df.groupby('Player')['BPM'].apply(list)
nba_bpm_lists = nba_bpm_lists.where(nba_bpm_lists.str.len() >= 4).dropna()

mean_nba_y3y4_bpm = (nba_bpm_lists.str[2] + nba_bpm_lists.str[3])/2

# Format player names for a compatible merge.
college_first_name = mean_college_season['Player ID'].str.split('/').str[1].str.split('-').str[0]
college_last_name = mean_college_season['Player ID'].str.split('/').str[1].str.split('-').str[1]
mean_college_season['name'] = college_first_name.str.cat(college_last_name, sep=' ')

mean_nba_y3y4_bpm = mean_nba_y3y4_bpm.reset_index()
mean_nba_y3y4_bpm['name'] = mean_nba_y3y4_bpm['Player'].str.replace('.', '').str.replace('*', '').str.replace("'", '')
mean_nba_y3y4_bpm.drop('Player', axis=1, inplace=True)

# Add target variable (y3y4 mean bpm) to mean_college_season.
mean_college_season = mean_college_season.merge(mean_nba_y3y4_bpm, how='left', on='name')

# A few names with a hyphen in the last name aren't merged in my solution. So we need to manually assign their BPM. 
# Luckily in this data set, there are only two: Chris Douglas-Roberts and Michael Kidd-Gilchrist.

# Douglas-Roberts:
mean_college_season.at[3223, 'BPM'] = -3.65

# Kidd-Gilchrist:
mean_college_season.at[11618, 'BPM'] = -1.1


# Drop column 'name'
mean_college_season.drop('name', axis=1, inplace=True)


# Write data to .csv file, so it can be used more efficiently in separate model script.
mean_college_season.to_csv('RealGMDataClean.csv')

