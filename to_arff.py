"""
Course: CSI5155
Tiffany Nien Fang Cheng
Group 33
Student ID: 300146741
"""
import json
import numpy as np
import pandas as pd
from scipy.io import arff as af
import arff
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from util import *

"""
The lib output the arff file and csv file after more data process
the data processing of this lib is to set the proper classification label
group_classes[0] to 1
group_classes[1] to 0
class1 is fatal injury
class2 is non-fatal injury
class3 is property damage only
"""
# grouping scenario for different cases
group_classes=[1,2]      # considering only injury cases
# group_classes = [1, (2, 3)]  # considering fatal injury vs the rest
# group_classes=[(1,2), 3] # considering injury vs property damage
# group_classes=[1,2,3]      # considering multiclass

# read the preprocessed file
with open("transportation_accident_mod1.csv", encoding="utf-8") as cvs_file:
    cvs_data = pd.read_csv(cvs_file, sep=',')

# taking the input and the label
Y = cvs_data['CLASS_OF_ACCIDENT']
X = cvs_data.copy(deep=True)
X.drop(['CLASS_OF_ACCIDENT', 'DATE', 'TIME', 'Month'], axis=1, inplace=True)
# X.drop(['CLASS_OF_ACCIDENT'], axis=1, inplace=True)
# convert true and false to 1 and 0
X['Highway'] = X['Highway'].astype(int)

# print(X.describe())
# x = X[['X']].values.astype(float)
# normalizaion for location value
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(X[['X']].values.astype("float64"))
X["X"] = pd.DataFrame(scaled_values)
scaled_values = scaler.fit_transform(X[['Y']].values.astype("float64"))
X["Y"] = pd.DataFrame(scaled_values)

# X["IMPACT_TYPE"] = X["IMPACT_TYPE"].replace({99:0})

# debug
# pd.options.display.max_columns = 500
# display(X.head(2))
# pd.options.display.max_columns = 0
# print(Y)
# print(X)

features = X.to_records(index=False).dtype

X = X.to_numpy()
Y = np.array(Y)
# print(features)

# re-label the class to according to different scenario
# put all the false class to -1 first, put the to be deleted classes to 0
# to put it in to 3 class still is to be able to reduce the size of the sample later
# using in class 1 vs class 2 scenario
if group_classes[0] is 1 and isinstance(group_classes[1], tuple):
    # scenario of class 1 vs class(2 and 3)
    Y[Y > 1] = -1
elif isinstance(group_classes[0], tuple):
    # scenario of class 3 vs class(1 and 2)
    Y[Y < 3] = 1
    Y[Y == 3] = -1
elif group_classes[0] is 1 and group_classes[1] is 2 and len(group_classes) is 2:
    # scenario of class 1 vs class 2
    Y[Y < 2] = 1
    Y[Y == 2] = -1
    Y[Y == 3] = 0
else:
    # multiclass scenario
    group_classes = [(1, 2, 3), 0]

# debug
# print(X)
# size reducing for class 1 vs class 2 and dropping class 3
X = np.delete(X, np.where(Y == 0), 0)
Y = np.delete(Y, np.where(Y == 0), 0)
# put the -1 label back to 0
Y = np.where(Y < 0, 0, Y)

# filling missing value from nan to 0
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imp = imp.fit(X)
traffic_control_cond = imp.transform(X)

# random_state=30
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=30)

print(len(X_train))
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)

# to get the list of features for output in the arff
features_variance = list()
tmp = list(features.descr)
print(tmp)
for i in range(len(tmp)):
    if tmp[i]:
        features_variance.append(tmp[i])
print(features_variance)
# features
FV = get_features(features_variance)
print(FV)

# make sure the classified result is in the shape of (n,1) or it's (n,)
Y = Y.reshape((len(Y), 1))
input_data_df_categorical_v = np.append(X, Y, axis=1)

# get arff obj
# get_arff_dump is in util.py
arff_dump_v = get_arff_dump(input_data_df_categorical_v, FV, "{}".format(group_classes))

#To generate arff and/or csv file
to_arff = True
to_csv = True
if to_arff:
    # name the file correctly
    with open("traffic_{}_{}.arff".format(group_classes[0], group_classes[1]), "w", encoding="utf8") as f:
        arff.dump(fp=f, obj=arff_dump_v)

if to_csv:
    with open("traffic_{}_{}.csv".format(group_classes[0], group_classes[1]), encoding="utf8", mode='w',
              newline='\n') as f:
        df = pd.DataFrame.from_records(input_data_df_categorical_v, columns=[x[0] for x in FV])
        df.to_csv(f, index=False)
