import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Dimensions that seem to be unnecessary: 4 6 7 (8?) (9?)
# Very important dimensions: 5, 13
IGNORED_DIMENSIONS = [4, 6, 7, 8]

# Include more sophisticated function to the input
def transform_data(data):
    logs = np.log10(data)
    log2s = np.log2(data)
    sqrts = np.sqrt(data)
    data = add_degrees(data)
    data = np.concatenate([data, logs, sqrts, log2s], axis=1)
    return data

# Adds additional degrees to the data.
def add_degrees(data):
    transformed = data
    for i in range(2, 8):
        new_degree = np.power(data, i)
        transformed = np.concatenate([transformed, new_degree], axis=1)
    return transformed

# Generates different regression techniques
def generate_regressors():
    clf_list = [linear_model.LassoLarsCV(), linear_model.LinearRegression(),
                linear_model.RidgeCV(), linear_model.ElasticNetCV(), svm.SVC(), ensemble.RandomForestClassifier()]
    clf_list = generate_lasso_regressors(clf_list)
    clf_list = generate_ridge_regressors(clf_list)
    clf_list = generate_decision_trees(clf_list)
    clf_list = generate_forest_classifiers(clf_list)
    return clf_list

# Generates LASSO regressors
def generate_lasso_regressors(regs):
    for i in np.linspace(0.5, 1, 10):
        regs.append(linear_model.Lasso(alpha=i))
    return regs

# Generates Ridge regressors
def generate_ridge_regressors(regs):
    for i in np.linspace(0.5, 1, 10):
        regs.append(linear_model.Ridge(alpha=i))
    return regs

# Generates Decision tree regressors
def generate_decision_trees(regs):
    for i in range(5, 10):
        regs.append(DecisionTreeRegressor(max_depth=i))
    return regs

# Generate Forest classifiers
def generate_forest_classifiers(regs):
    for i in range(5, 10):
        regs.append(ensemble.RandomForestClassifier(max_depth=i))
    return regs

# Load the training data
file_data = open("data/train.csv","rb")
data = np.loadtxt(file_data, delimiter=",")

# Extract ids, target and characteristics
ids = data[:, 0]
y = data[:, 15]
data = data[:, 1:15]

# Turns the data into a more complex function and
# ignore less important dimensions
data = scipy.delete(data, IGNORED_DIMENSIONS, 1)
X = transform_data(data)

# Create a list of classifier candidates
clf_list = generate_regressors()

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
X_test = scipy.delete(X_test, IGNORED_DIMENSIONS, 1)
X_test = transform_data(X_test)
y_test = best_classifier.predict(X_test)
pd.DataFrame({'Id': ids.astype(int), 'Delay': y_test}).to_csv("out.csv", index=False, columns=['Id', 'Delay'])

