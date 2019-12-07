"""
Course: CSI5155
Tiffany Nien Fang Cheng
Group 33
Student ID: 300146741
"""
import json
import numpy as np
import pandas as pd
from pprint import pprint as pp
import matplotlib.pyplot as plt
from collections import Counter

# read the raw data
with open("transportation_accident.csv",  encoding="utf-8") as csv_file:
    cvs_data = pd.read_csv(csv_file, sep=',')

# debugging display
# pd.options.display.max_columns = 500
# print(cvs_data.head(2))

# del the following features since they are dup to each others or meaningful only to db
new_cvs = cvs_data.drop(columns=["ANOM_ID","YEAR","GEO_ID","LONGITUDE", "LATITUDE", "ObjectId"])

# Extracting LOCATION info to hwy or not
# TODO: can do one hot of different road types: road, ave, street and drive
hwy_binary_list = list()
# print(new_cvs['LOCATION'].dtypes, new_cvs['LOCATION'].name)
hwy_binary_list = new_cvs['LOCATION'].str.contains("highway|hwy", case=False)
# pp(hwy_binary_list.__dict__)
hwy_binary_list.name = 'Highway'
new_cvs['Highway'] = hwy_binary_list
# drop detailed location after extracting highway info
new_cvs = new_cvs.drop(columns=["LOCATION"])

# date string to datatime obj handling
temp_date = new_cvs['DATE']
temp_date = pd.to_datetime(temp_date).dt.date
# pp(temp_date.__dict__)
# print(temp_date)
new_cvs['DATE'] = temp_date
# One hot for 4 seasons
new_cvs['Month'] = pd.to_datetime(temp_date).dt.month_name()
season_dict = dict(January='Winter', February='Winter', March='Spring', April='Spring', May='Spring',
                   June='Summer', July='Summer', August='Summer', September='Fall', October='Fall',
                   November='Fall', December='Winter')
new_cvs['Seasons'] = new_cvs['Month'].replace(season_dict)



# time string to datetime obj handling
# TODO: maybe categorize into different time of day 0AM-6AM Rank 4,
#  12PM to 6 PM rank 3 or one hot into 4 time slots
# print("********************************************************************")
temp_time = new_cvs['TIME']
# dealing with missing timestamp
unknown_time = new_cvs['TIME'].str.contains("unknown", case=False)
temp_time = new_cvs['TIME'].replace("Unknown", pd.NaT)

# period obj is not working here
# temp_time = pd.to_datetime(temp_time, format='%I:%M:%S %p').dt.to_period(freq="H")
# temp_time.to_period(freq="H")

# taking time only from the datetime obj
temp_time = pd.to_datetime(temp_time, format='%I:%M:%S %p').dt.time
new_cvs['TIME'] = temp_time
# not working yet
# test = new_cvs['TIME'].between_time("0:00", "6:00", include_start=True, include_end=False)


# Extracting the rank from a mix of int and string out of those columns
rank_to_int_list = ["TRAFFIC_CONTROL", "TRAFFIC_CONTROL_CONDITION", "ROAD_SURFACE_CONDITION", "LIGHT", "ENVIRONMENT",
                    "IMPACT_TYPE", "CLASS_OF_ACCIDENT", "ACCIDENT_LOCATION"]
for feature in rank_to_int_list:
    new_cvs[feature] = new_cvs[feature].str.split(expand=True)
    pp(new_cvs[feature] .__dict__)

# debug
# print(new_cvs.loc[133])

# the above features all have unknown or others, treating them as missing data and set to 0
for column in ['IMPACT_TYPE', 'ENVIRONMENT', 'LIGHT', 'ROAD_SURFACE_CONDITION']:
    new_cvs[column] = new_cvs[column].replace({99: 0, "99": "0", 98: 0, "98": "0"})

# 10 is unknown in traffic_control set to 0
new_cvs['TRAFFIC_CONTROL'] = new_cvs['TRAFFIC_CONTROL'].replace({10: 0, '10': "0"})

# One hot for traffic_control and accident_location and seasons
categories = ["TRAFFIC_CONTROL", "ACCIDENT_LOCATION", "Seasons"]
for col in categories:
    prefix = ('categorical_'+col)
    dummies = pd.get_dummies(new_cvs[col], prefix=prefix, dtype='int8')

    test = new_cvs[col].value_counts()
    # drop the others or unknown type of column
    if "TRAFFIC_CONTROL" in col:
        dummies.drop((prefix + '_0'), inplace=True, axis=1)
    elif "ACCIDENT_LOCATION" in col:
        # remove noise
        dummies.drop((prefix + '_98'), inplace=True, axis=1)

    new_cvs.drop(col, inplace=True, axis=1)
    new_cvs = pd.concat([new_cvs, dummies], axis=1)

# debug
# pd.options.display.max_columns = 0
cvs_np = cvs_data.to_records()

# save to csv file
with open("transportation_accident_mod1.csv", mode='w', newline='\n') as f:
    new_cvs.to_csv(f, index=False)

def draw_his(lib_call='counts'):
    """
    The functoin is to generate the his from the pre-processed data
    :param lib_call: draw_his from pandas value_counts(counts, default) or matplotlib histogram (his)
    :return:
    """
    for key in new_cvs.head():
        # the following info has no histogram to display
        if key in ["YEAR","ANOM_ID", 'DATE', "TIME", "X", "Y", "TRAFFIC_CONTROL_CONDITION",
                   "LOCATION", "GEO_ID", "LONGITUDE",	"LATITUDE",	"ObjectId"]:
            continue
        if "counts" in lib_call:
            # using value_counts lib
            plt.figure()
            x = new_cvs[key]
            # x = Counter(new_cvs[key])
            n = pd.Series(x).value_counts()
            n.plot(kind='bar')
            for i in range(len(n)):
                print("{}, {}".format(i, n[i]))
                text = plt.text(i, n[i], str(n[i]))
                print(text)
        elif 'his' in lib_call:
            # using histogram lib
            x = new_cvs[key]
            n, bins, patches = plt.hist(x)
            for i in range(len(n)):
                if n[i] <1:
                    continue
                plt.text(bins[i], n[i], str(n[i]))

        plt.title(key)
        # plt.show()

        plt.savefig('{}_his.png'.format(key))
        plt.close()

# generating histograms
draw_his()


