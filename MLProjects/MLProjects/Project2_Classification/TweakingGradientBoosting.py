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
    y = data[:, 8]
    X = data[:, 1:8]


    # Create the pipeline
    pipeline = Pipeline([#("feature_selection", SelectKBest(score_func=f_regression)), #not working and not interesting
                         ("scale", MinMaxScaler()), #standardscaler does not work
                         ("classify", GradientBoostingClassifier())])


    # Create a list of potential parameters (sigmoid kernel nonsense, only introduced for neural networks)
    params = { 
        loss : ['deviance', 'exponential'], # default=’deviance’
        learning_rate : [0.1,0.5,1, 10], # default=0.1
        n_estimators : [100], # default = 100
        max_depth : [3], # default = 3
        min_samples_split : [2], # default = 2
        min_samples_leaf : [1], # default = 1
        min_weight_fraction_leaf : [0], # default = 0
        subsample : [1.0], # default = 1.0
        max_features : [None], # default = None
        max_leaf_nodes : [None], # default = None
        init : [None], # default = None
        warm_start : [False], # default = False
        random_state : [None], # default = None
        presort : ['auto'] # default = 'auto'
    }

    # Grid search to find best parameters
    gs = GridSearchCV(pipeline, params, verbose=2, refit=True, cv=5, n_jobs = -1)
    gs.fit(X, y)
    pipeline.set_params(**gs.best_params_)
    
    # Output results
    gridSearchOverviewfile = open(gridSearchOverviewFile, 'w+')
    scores = gs.grid_scores_
    scores = sorted(scores, reverse=True, key=lambda x: x[1]) #mean_validation_score at position 1
    for curScore in scores:
        gridSearchOverviewfile.write("\n" + str(curScore[1]) + " --- " + str(curScore[0])) #parameters at position 0
    test_error = np.mean(cross_validation.cross_val_score(pipeline, X, y, cv=5, scoring='accuracy', n_jobs = -1))
    print "BEST SOLUTION:"
    print gs.best_params_, ': \n -->', test_error, '\n \n'


    # Do the prediction of the test and validation data with best configuration
    file_test_data = open("Project2_Classification/data/validate_and_test.csv","rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:8]
    y_test = gs.predict(X_test)
    y_test = [int(y) for y in y_test]
    pd.DataFrame({'Id': ids.astype(int), 'Label': y_test}).to_csv("classification_prediction.csv", index=False, columns=['Id', 'Label'])