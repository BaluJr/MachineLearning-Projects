import numpy as np
from sklearn.pipeline import Pipeline
# preprocessin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
# feature selection
from sklearn.feature_selection import *
from sklearn.cross_validation import StratifiedKFold
# estimation
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model, cross_validation, svm, ensemble, gaussian_process
from sklearn.tree import DecisionTreeRegressor
# evaluation
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
#output
import pandas as pd




if __name__ == "__main__":

    # Load the training data
    file_data = open("Project1_LinearRegression/data/train.csv","rb")
    data = np.loadtxt(file_data, delimiter=",")
    ids = data[:, 0]
    y = data[:, 15]
    y = np.log(y)
    X = data[:, 1:15]

    #log all the x values, which are 2 based
    dims_to_log = [7,8,10,11,12]
    for dim in dims_to_log:
        X[:,dim] = np.log2(X[:,dim])


    # Create the pipeline
    pipeline = Pipeline([("feature_selection", SelectKBest(score_func=f_regression)),
                         ("scaling", MinMaxScaler()),
                         ("regression", ensemble.RandomForestRegressor())])


    # Create a list of potential parameters
    # see http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    parameters = {
        'feature_selection__k': [6,7],
        'regression__n_estimators': [int(i) for i in np.linspace(42,46, 2)],
        'regression__criterion': ["mse"],
        'regression__max_depth': [6],
        'regression__max_features': [0.9],
        'regression__min_samples_split': [3,4],
        'regression__min_samples_leaf': [3,4,5],
        'regression__min_weight_fraction_leaf': np.linspace(0.011, 0.011, num=1),
        'regression__max_leaf_nodes': [None],
        'regression__bootstrap': [True],
        'regression__oob_score': [False],
        'regression__n_jobs': [-1],
        'regression__random_state': [None,1], 
        'regression__verbose': [0],
        'regression__warm_start': [False]
    }


    # Grid search for best parameters
    gs = GridSearchCV(pipeline, parameters, verbose=2, refit=False, cv=6, n_jobs = -1)
    gs.fit(X, np.sqrt(y))
    pipeline.set_params(**gs.best_params_)
    test_error = np.mean(np.sqrt(np.abs(cross_validation.cross_val_score(pipeline, X, y, cv=6, scoring='mean_squared_error'))))
    print "\n\n Best Solution:"
    print pipeline, ': \n -->', test_error
    pipeline.fit(X, y)    


    # Do the prediction of the test and validation data with best configuration
    file_test_data = open("Project1_LinearRegression/data/validate_and_test.csv","rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:15]
    for dim in dims_to_log:
        X_test[:,dim] = np.log2(X_test[:,dim])
    y_test = pipeline.predict(X_test)
    y_test = np.exp(y_test)
    pd.DataFrame({'Id': ids.astype(int), 'Delay': y_test}).to_csv("bestout.csv", index=False, columns=['Id', 'Delay'])