import numpy as np
from sklearn.pipeline import Pipeline
# preprocessin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
# feature selection
from sklearn.feature_selection import *
from sklearn.cross_validation import StratifiedKFold
#classification
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# evaluation
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
#output
import pandas as pd

#CONFIG
gridSearchOverviewFile = "GridSearch.csv"

if __name__ == "__main__":

    # Load the training data
    file_data = open("Project2_Classification/data/train.csv","rb")
    data = np.loadtxt(file_data, delimiter=",")
    ids = data[:, 0]
    Y = data[:, 8]
    X = data[:, 1:8]

    # Load the test data
    file_test_data = open("Project2_Classification/data/validate_and_test.csv","rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:8]



    ### USE THREE DIFFERENT CLASSIFIERS AND TAKE THE BEST RESULT
    # 1 ---------------
    pipeline1 = Pipeline([#("feature_selection", SelectKBest(score_func=f_regression)), #not working and not interesting
                         ("scale", MinMaxScaler()), #standardscaler does not work
                         ("classify", SVC())])
    # Create a list of potential parameters (sigmoid kernel nonsense, only introduced for neural networks)
    param_grid = [
      #{ 'classify__kernel': ['poly'],'classify__degree': [2,3,4,5]},
      { 'classify__kernel': ['rbf'] },
     ]
    # Mix common parameters into all grids
    for param in param_grid: 
        param.update({
            #'feature_selection__k': [7],
            'classify__C': [67],
            'classify__gamma': [1.1],
            'classify__tol': [0.1],
            'classify__cache_size': [4096] 
        })
    # Grid search to find best parameters
    gs1 = GridSearchCV(pipeline1, param_grid, verbose=2, refit=True, cv=5, n_jobs = -1)
    gs1.fit(X, Y)
    pipeline1.set_params(**gs1.best_params_)
    # Predict with first classifier
    y1 = [int(y) for y in gs1.predict(X_test)]


    # 2 ---------------
    pipeline2 = Pipeline([#("feature_selection", SelectKBest(score_func=f_regression)), #not working and not interesting
                         ("scale", MinMaxScaler()), #standardscaler does not work
                         ("classify", RandomForestClassifier())])
    # Create a list of potential parameters (sigmoid kernel nonsense, only introduced for neural networks)
    params = {
        'classify__n_estimators': [70], 
        'classify__warm_start': [False], 
        'classify__criterion': ['gini'], 
        'classify__max_leaf_nodes': [None], 
        'classify__bootstrap': [True], 
        'classify__min_samples_split': [3], 
        'classify__n_jobs': [-1], 
        'classify__oob_score': [False], 
        'classify__min_weight_fraction_leaf': [0.01], 
        'classify__random_state': [1], 
        'classify__max_features': [None], 
        'classify__verbose': [0], 
        'classify__min_samples_leaf': [3], 
        'classify__max_depth': [12]
    }
    # Grid search to find best parameters
    gs2 = GridSearchCV(pipeline2, params, verbose=2, refit=True, cv=5, n_jobs = -1)
    gs2.fit(X, Y)
    pipeline2.set_params(**gs2.best_params_)
    # Predict with second classifier
    y2 = [int(y) for y in gs2.predict(X_test)]
    


    ## 3 ---------------
    #pipeline3 = Pipeline([#("feature_selection", SelectKBest(score_func=f_regression)), #not working and not interesting
    #                     ("scale", MinMaxScaler()), #standardscaler does not work
    #                     ("classify", GradientBoostingClassifier())])
    ## Create a list of potential parameters (sigmoid kernel nonsense, only introduced for neural networks)
    #param_grid = [
    #  { 'classify__kernel': ['poly'],'classify__degree': [2,3,4,5]},
    # ]
    ## Mix common parameters into all grids
    #for param in param_grid: 
    #    param.update({
    #        'feature_selection__k': [7],
    #        'classify__C': np.linspace(60, 80, 21),
    #        'classify__gamma': [0.8,0.9,1,1.1,1.2],
    #        'classify__tol': [1,0.1,1e-2,1e-3,1e-4],
    #        'classify__n_estimators': [4096],
    #    })
    ## Grid search to find best parameters
    #gs3 = GridSearchCV(pipeline3, {}, verbose=2, refit=True, cv=5, n_jobs = -1)
    #gs3.fit(X, Y)
    #pipeline3.set_params(**gs3.best_params_)
    ## Predict with third classifier
    #y3 = [int(y) for y in gs3.predict(X_test)]


    # Determine the best approximation out of the given three estimators
    y_final = []
    for i in xrange(0, len(y1)):
        if y1[i] != y2[i]: # != y3[i]:
            y_final.append("UNSURE")
        #elif y1[i] == y2[i] == y3[i]:
        #    y_final.append("SURE")
        else:
            lst = [y1[i], y2[i]] #, y3[i]]
            y_final.append(max(set(lst), key=lst.count))

    pd.DataFrame({'Id': ids.astype(int), 'Label': y_final}).to_csv("classification_prediction.csv", index=False, columns=['Id', 'Label'])