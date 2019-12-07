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
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import ClusterCentroids, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from util import *

# grouping scenario for different cases
group_classes=[1,2]      # considering only injury cases
# group_classes = [1, (2, 3)]  # considering fatal injury vs the rest
# group_classes = [(1, 2), 3]  # considering injury vs property damage
# group_classes=[(1,2,3),0]      # considering multiclass

with open("traffic_{}_{}.csv".format(group_classes[0], group_classes[1]), encoding="utf8") as f:
    cvs_data = pd.read_csv(f, sep=',')

# seperate the input and the classification result
Y = cvs_data['class']
X = cvs_data.copy(deep=True)
# X.drop(['class', "TIME", "DATE"], axis=1, inplace=True)
X.drop(['class'], axis=1, inplace=True)
import ipdb
ipdb.set_trace()


# convert pandas to numpy
features = X.to_records(index=False).dtype
X = X.to_numpy()
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

print(len(X_train))
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)

# replacing missing data nan to 0
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imp = imp.fit(X)
X = imp.transform(X)


def sampling(X, Y, sample_type="over"):
    """
    This is to pick the sampling technique and output the data after sampled
    :param X: input data
    :param Y: classification data
    :param sample_type: can take a list or str of sampling technique
                        default is oversampling. options: over, under, combine
    :return: cascade data of X and Y
    """
    if "over" in sample_type:
        # using SMOTE for over sampling
        X_oversampled, y_oversampled = SMOTE(sampling_strategy="minority", random_state=42).fit_resample(X, Y)
    if "under" in sample_type:
        # using ENN for under sampling, since centroid has memory issues
        # centroid undersample
        # X_under, y_under = ClusterCentroids(random_state=42).fit_resample(X,Y)
        X_under, y_under = EditedNearestNeighbours(random_state=42).fit_resample(X, Y)
    if "combine" in sample_type:
        # using sklearn built-in SMOTEENN for comebined sampling
        # because centroids has memory issue
        X_comb, y_comb = SMOTEENN(random_state=42).fit_resample(X, Y)
        # X_oversampled, y_oversampled = SMOTE(sampling_strategy="minority", random_state=42).fit_resample(X, Y)
        # X_comb, y_comb = ClusterCentroids(random_state=42).fit_resample(X_oversampled,y_oversampled)

    X_Y_under = list()
    X_Y_over = list()
    X_Y_comb = list()
    X_Y = dict()
    # append the data back for return
    if 'under' in sample_type:
        X_Y_under = np.append(X_under, y_under.reshape(len(y_under), 1), axis=1)
    if 'over' in sample_type:
        X_Y_over = np.append(X_oversampled, y_oversampled.reshape(len(y_oversampled), 1), axis=1)
    if 'combine' in sample_type:
        X_Y_comb = np.append(X_comb, y_comb.reshape(len(y_comb), 1), axis=1)

    X_Y.setdefault("under", X_Y_under)
    X_Y.setdefault("over", X_Y_over)
    X_Y.setdefault("combine", X_Y_comb)
    return X_Y


# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# get the name and data type of features
features_variance = list()
tmp = list(features.descr)
print(tmp)
for i in range(len(tmp)):
    if tmp[i]:
        features_variance.append(tmp[i])
print(features_variance)
# features
feature_list = get_features(features_variance)


def feature_selection(X, Y, FV, estimators=100):
    """
    Using tree as feature selection instead of variance
    because number of pedestrian is very import feature but there are a lot of zeros in the feature
    :param X: Input data
    :param Y: Label
    :param FV: list of features
    :param estimators: estimator passing to the tree algo
    :return: reduced data and feature list
    """
    # using ExtraTreesClassifier for getting the important features
    clf = ExtraTreesClassifier(n_estimators=estimators)
    clf = clf.fit(X, Y)
    # feature selection
    clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    features_lsv = model.get_support()
    # get the name of the selected features
    feature_list = list()
    for i in range(len(features_lsv)):
        if features_lsv[i]:
            feature_list.append(FV[i])

    # debug
    # print the feature selected out
    # [print(FV[i]) for i in range(len(features_lsv)) if features_lsv[i]]

    # to reduce the column of X
    X_new = model.transform(X)
    # X_new.shape

    # append class label back to data
    feature_list.append(FV[-1])
    X_Y = np.append(X_new, Y.reshape(len(X_new), 1), axis=1)
    return X_Y, feature_list

# file control
select_feature = False
SAMPLE = "over"

if select_feature:
    X_Y, feature_list = feature_selection(X, Y, feature_list)

if SAMPLE:
    X_Y = sampling(X, Y, SAMPLE)[SAMPLE]

# get the arff obj
arff_dump_v = get_arff_dump(X_Y, feature_list, "selection")

# append the file name properly
file_name_list=[group_classes[0], group_classes[1]]
if select_feature:
    file_name_list.append("selection")
elif SAMPLE:
    file_name_list.append(SAMPLE)
file_name='traffic'
for it in file_name_list:
    file_name = "{}_{}".format(file_name,it)

# generate feature selectiong arff or sampling arff
with open("{}.arff".format(file_name), "w", encoding="utf8") as f:
    arff.dump(fp=f, obj=arff_dump_v)
