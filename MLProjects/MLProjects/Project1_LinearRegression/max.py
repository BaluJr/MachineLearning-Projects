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
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import *


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
    transformed = np.insert(transformed, 0, values=1, axis=1)
    return transformed

# Adds combination of features
def combine(data):
    combinations = data
    for i in range (0, 1):
        new_combination = data * data[i]
        combinations = np.concatenate([combinations, new_combination], axis=1)
    return combinations




# MAIN ---------------------------------------------------------------------
if __name__ == "__main__":
    # Load the training data
    file_data = open("Project1_LinearRegression/data/train.csv","rb")
    data = np.loadtxt(file_data, delimiter=",")

    # Extract ids, target and characteristics
    ids = data[:, 0]
    y = data[:, 15]
    y = np.log(y)
    data = data[:, 1:15]

    # Optimize imput data
    fs=SelectKBest(score_func=f_regression,k=5)
    X = fs.fit_transform(data, y)
    ignored_dimensions = [i for i in range(0, 14) if fs.get_support()[i]==False]
    scalerX = MinMaxScaler().fit(X)
    X = scalerX.transform(X)
    #X = transform_data(X)



    # Select the Regressor
    clf = ensemble.RandomForestRegressor()
    
    # Create a list of potential parameters
    # see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    parameters = {
        'n_estimators': [int(x) for x in np.logspace(1,3,10)],
        'criterion': ["mse"],
        'max_depth': np.linspace(2, 11, 10),
        'max_features': [None, 0.7, 0.8, 0.9],
        'min_samples_split': [2,3,4],
        'min_samples_leaf': [1,2,3],
        'min_weight_fraction_leaf': np.linspace(0., .1, num=10),
        'max_leaf_nodes': [None],
        'bootstrap': [True],
        'oob_score': [False],
        'n_jobs': [-1],
        'random_state': [None], 
        'verbose': [0],
        'warm_start': [False]
    }

    # Grid search for best parameters
    gs = GridSearchCV(clf, parameters, verbose=2, refit=False, cv=6, n_jobs = -1) # n_jobs=-1 -> use all cpu cores
    gs.fit(X, y)
    clf.set_params(**gs.best_params_)
    test_error = np.mean(np.sqrt(np.abs(cross_validation.cross_val_score(clf, X, y, cv=6, scoring='mean_squared_error'))))
    print "\n\n Best Solution:"
    print clf, ': \n -->', test_error
    clf.fit(X, y)    
    
    

    # Do the prediction of the test and validation data with best configuration
    file_test_data = open("Project1_LinearRegression/data/validate_and_test.csv","rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:15]
    X_test = np.delete(X_test, ignored_dimensions, axis=1)
    X_test = scalerX.transform(X_test)
    #X_test = transform_data(X_test)
    y_test = clf.predict(X_test)
    y_test = np.exp(y_test)
    pd.DataFrame({'Id': ids.astype(int), 'Delay': y_test}).to_csv("out.csv", index=False, columns=['Id', 'Delay'])