import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LassoCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 

# Use the number of bins
INTENSITY_BINS = 50
SEGMENTED_INTENSITY = False
FCC = False


if __name__ == "__main__":

    # Load data from the train file
    file_data = open("Project3_FeatureEngineering/data/train" + ("_segbins" if SEGMENTED_INTENSITY else "_bins") + str(INTENSITY_BINS) + ("_fcc.csv" if FCC else ".csv"), "rb")
    data = np.loadtxt(file_data, delimiter=",")

    # Take only the relevant information
    Y = data[:, 697 + INTENSITY_BINS]
    X = data[:, 1:697 + INTENSITY_BINS]

    # Use ExtraTreesClassifier
    pipeline = Pipeline([("scaling", MinMaxScaler()),
                         ("feature_selection", SelectKBest()),
                         ("feature_selection2", PCA()),
                        ("classify", ExtraTreesClassifier())])
    params = {
            'feature_selection__k':  np.linspace(50, 150, 11), # for SelectKBest
            'feature_selection2__n_components': [10, 20, 30, 40, 50], # for PCA
            'classify__bootstrap': [False], 
            'classify__class_weight': [None], 
            'classify__criterion': ['gini'], 
            'classify__max_depth': [None],
            'classify__max_features': [None], 
            'classify__max_leaf_nodes': [None], 
            'classify__min_samples_leaf': [4,5],
            'classify__min_samples_split': [3,4], 
            'classify__min_weight_fraction_leaf': [0.2], 
            'classify__n_estimators': [40,50,60],
            'classify__n_jobs': [-1], 
            'classify__oob_score': [False], 
            'classify__random_state': [None], 
            'classify__verbose': [0],
            'classify__warm_start': [True]
        }

    # Grid search to find best parameters
    gs1 = GridSearchCV(pipeline, params, verbose=2, refit=True, cv=10, n_jobs = -1)
    gs1.fit(X, Y)
    print "\n\nBest Score: ", gs1.best_score_
    print "\nWith Parameters: ", gs1.best_params_
    pipeline.set_params(**gs1.best_params_)
 

    # Load the validation and test data and predict the result
    file_test_data = open("Project3_FeatureEngineering/data/test_validate" + ("_segbins" if SEGMENTED_INTENSITY else "_bins") + str(INTENSITY_BINS) + ("_fcc.csv" if FCC else ".csv"), "rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:697 + INTENSITY_BINS]
    y_test = gs1.predict(X_test)


    # Write output
    pd.DataFrame({'Id': ids.astype(int), 'Label': y_test.astype(int)})\
        .to_csv("Project3_FeatureEngineering/results/out"+  ("_segbins" if SEGMENTED_INTENSITY else "_bins") + str(INTENSITY_BINS) + ("_fcc.csv" if FCC else ".csv"), index=False, columns=['Id', 'Label'])
