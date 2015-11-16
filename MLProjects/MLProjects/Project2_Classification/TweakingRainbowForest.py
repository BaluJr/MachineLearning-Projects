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
from sklearn.ensemble import RandomForestClassifier
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
    pipeline = Pipeline([#("feature_selection", SelectKBest(score_func=f_classify)), #not working and not interesting
                         ("scale", MinMaxScaler()), #standardscaler does not work
                         ("classify", RandomForestClassifier())])


    # Create a list of potential parameters (sigmoid kernel nonsense, only introduced for neural networks)
    params = {
        'classify__n_estimators': [int(i) for i in np.linspace(62, 78, 5)],
        'classify__criterion': ["gini"],
        'classify__max_features': [None],
        'classify__max_depth': [int(i) for i in np.linspace(10, 20, 6)],
        'classify__min_samples_split': [2,3,4],
        'classify__min_samples_leaf': [2,3,4],
        'classify__min_weight_fraction_leaf': np.linspace(0.005, 0.015, 6),
        'classify__max_leaf_nodes': [None],
        'classify__bootstrap': [True],
        'classify__oob_score': [False],
        'classify__n_jobs': [-1],
        'classify__random_state': [None], 
        'classify__verbose': [0],
        'classify__warm_start': [False]
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