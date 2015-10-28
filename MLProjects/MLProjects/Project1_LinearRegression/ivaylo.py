import numpy as np
import pandas as pd
from sklearn import linear_model, cross_validation, svm, ensemble
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble
import itertools
from sklearn import gaussian_process
from sklearn.svm import SVR

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
    for i in range(4, 10):
        new_degree = np.power(data, i)
        transformed = np.concatenate([transformed, new_degree], axis=1)
    return transformed



# MAIN -----------------------------------------------------------
# Load the training data
file_data = open("Project1_LinearRegression/data/train.csv","rb")
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

# Create a list of regressor candidates
clf_list = [ensemble.RandomForestClassifier()]
                #DecisionTreeRegressor()
                #linear_model.LassoLarsCV(), linear_model.LinearRegression(), DecisionTreeRegressor(max_depth=6),
                #linear_model.RidgeCV(), linear_model.ElasticNetCV(), , 
                #linear_model.BayesianRidge(compute_score=True), ensemble.ExtraTreesRegressor(n_estimators=10,random_state=42)]

# Create a list of potential paramet
list_of_parameter_sets = np.array([
                   ['max_depth', [6,7,8,9]],
                   ['min_samples_split', np.linspace(1, 3, num=3)],
                   ['min_samples_leaf', np.linspace(6, 13, num=7)],
                   ['min_weight_fraction_leaf', np.linspace(0., .4, num=4)],
                   ['max_features', [0.7, 0.8, 0.9]]
                 ],dtype=object)
best_classifier, best_score, best_params = None, 2000, None

# evaluate regressors with 6-fold cross validation, retain the best one
for clf in clf_list:
    # go through all parameter sets
    for parameter_set in list(itertools.product( *list_of_parameter_sets[:,1]) ):
        # set the parameters
        for i,parameter in enumerate(parameter_set):
            clf.set_params(**{list_of_parameter_sets[:,0][i]:parameter})

        # Do the test with the current parameter configuration
        clf.fit(X, y)    
        test_error = np.mean(np.sqrt(np.abs(cross_validation.cross_val_score(clf, X, y, cv=6, scoring='mean_squared_error'))))
        if test_error < best_score:
            best_score, best_classifier, best_params = test_error, clf, parameter_set
   
    # Output for regressor
    #print 'Train error for ', clf, ': ', np.sqrt(mean_squared_error(clf.predict(X), y))
    print '\n', clf, ':\n'
    #print 'Parameters: ', zip(list_of_parameter_sets[:,0], best_params), '\n'
    print 'Test error: ', test_error, '\n\n'


# Predict with the best regressor on test data and write them to out.csv
print "\n\n###################\nBest classifier is " + str(best_classifier) + " with test error: " + str(best_score) + "\n"
best_classifier.fit(X, y)

# Read test and validation data
file_test_data = open("Project1_LinearRegression/data/validate_and_test.csv","rb")
test_data = np.loadtxt(file_test_data, delimiter=",")
ids = test_data[:, 0]
X_test = test_data[:, 1:15]
X_test = np.delete(X_test, ignored_dimensions, axis=1)
X_test = scalerX.transform(X_test)
#X_test = transform_data(X_test)
y_test = best_classifier.predict(X_test)
pd.DataFrame({'Id': ids.astype(int), 'Delay': y_test}).to_csv("out.csv", index=False, columns=['Id', 'Delay'])
