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
from pprint import pprint as pp
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

"""
The lib here is proof of concept of doing training in python
but the main running model and getting accuracy result is running from weka
"""

# grouping scenario for different cases
# please check to_arff for detail explain
# group_classes=[1,2]
# group_classes=[1,(2,3)]
# group_classes=[(1,2),3]
group_classes=[(1,2,3),0]

with open("traffic_{}_{}.csv".format(group_classes[0], group_classes[1]), encoding="utf8") as f:
    cvs_data = pd.read_csv(f, sep=',')

Y = cvs_data['class']
X = cvs_data.copy(deep=True)
X.drop(['class'], axis=1, inplace=True)

# print(X.describe())
# x = X[['X']].values.astype(float)
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(X[['X']].values.astype("float64"))
X["X"] = pd.DataFrame(scaled_values)
scaled_values = scaler.fit_transform(X[['Y']].values.astype("float64"))
X["Y"] = pd.DataFrame(scaled_values)

# debug
# pd.options.display.max_columns = 500
# display(X.head(2))
# pd.options.display.max_columns = 0
# print(Y)
# print(X)

features = X.to_records(index=False).dtype
X = X.to_numpy()
Y = np.array(Y)
print(features)

imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
imp = imp.fit(X)
X = imp.transform(X)

# random_state=42
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=30)

print(len(X_train))
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', Y_test.shape)

# Emsemble based
# Instantiate model with 1000 decision trees
# random_state=42
rf = RandomForestClassifier(n_estimators=1000)
# Train the model on training data
rf.fit(X_train, Y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
print(predictions)

result = np.where(Y_test == 1)
print(Y_test[result])

print("Random Forest Tree")
a = confusion_matrix(Y_test, predictions)
# a = multilabel_confusion_matrix(Y_test, predictions)
display(a)


# neighbor based
n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
result = np.where(Y_test == 1)
print("KNN")
a = confusion_matrix(Y_test, predictions)
display(a)

# rule based
rule = DummyClassifier()
rule.fit(X_train, Y_train)
predictions = clf.predict(X_test)
result = np.where(predictions == 1)
print("Dummy rule")
print(result)
a = confusion_matrix(Y_test, predictions)
display(a)

# linear based
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, Y_train)
predictions = lin_clf.predict(X_test)
result = np.where(predictions == 1)
print("Linear SVM")
print(result)
a = confusion_matrix(Y_test, predictions)
display(a)

# tree based
tree = DecisionTreeClassifier()
tree.fit(X_train,Y_train)
predictions = tree.predict(X_test)
result = np.where(predictions == 1)
print("Tree DT")
print(result)
a = confusion_matrix(Y_test, predictions)
display(a)


