import numpy as np
import pandas as pd
from sklearn import linear_model, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Include more sophisticated function to the input
def transform_data(features):
    #logs = np.log10(data)
    #log2s = np.log2(data)
    #sqrts = np.sqrt(data)
    features = add_degrees(features)
    #data = np.concatenate([data, logs, sqrts, log2s], axis=1)
    return features

# Adds additional degrees to the data.
def add_degrees(data):
    transformed = data
    for i in range(2, 8):
        new_degree = np.power(data, i)
        transformed = np.concatenate([transformed, new_degree], axis=1)
    return transformed

# Load the training data
file_data = open("data/train.csv","rb")
data = np.loadtxt(file_data, delimiter=",")

# Extract ids, target and characteristics
ids = data[:, 0]
y = data[:, 15]
data = data[:, 1:15]

from sklearn.feature_selection import *
fs=SelectKBest(score_func=f_regression,k=5)
X = fs.fit_transform(data, y)
ignored_dimensions = [i for i in range(0, 14) if fs.get_support()[i]==False]

scalerX = StandardScaler().fit(X)
X = scalerX.transform(X)
#X = transform_data(X)

# Create a list of classifier candidates
clf_list = [linear_model.LassoLarsCV(), linear_model.LinearRegression(), DecisionTreeRegressor(max_depth=6),
                linear_model.RidgeCV(), linear_model.ElasticNetCV(), ensemble.RandomForestClassifier(),
                DecisionTreeRegressor(max_depth=5), DecisionTreeRegressor(max_depth=7)]

best_classifier, best_score = None, 2000

# evaluate classifiers with 6-fold cross validation, retain the best one
for clf in clf_list:
    clf.fit(X, y)
    print 'Train error for ', clf, ': ', np.sqrt(mean_squared_error(clf.predict(X), y))

    test_error = np.mean(np.sqrt(np.abs(cross_validation.cross_val_score(clf, X,
                                                                         y, cv=6, scoring='mean_squared_error'))))
    print 'Test error: ', test_error

    if test_error < best_score:
        best_score, best_classifier = test_error, clf

# Predict with the best classifiers on test data and write them to out.csv
print "Best classifier is " + str(best_classifier) + " with test error: " + str(best_score)
best_classifier.fit(X, y)

# Read test and validation data
file_test_data = open("data/validate_and_test.csv","rb")
test_data = np.loadtxt(file_test_data, delimiter=",")
ids = test_data[:, 0]
X_test = test_data[:, 1:15]
X_test = np.delete(X_test, ignored_dimensions, axis=1)
X_test = scalerX.transform(X_test)
#X_test = transform_data(X_test)
y_test = best_classifier.predict(X_test)
pd.DataFrame({'Id': ids.astype(int), 'Delay': y_test}).to_csv("out.csv", index=False, columns=['Id', 'Delay'])
