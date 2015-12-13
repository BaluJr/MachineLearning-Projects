import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

if __name__ == "__main__":

    # Load data from the train file
    file_data = open("Project3_FeatureEngineering/data/train.csv", "rb")
    data = np.loadtxt(file_data, delimiter=",")

    # Take only the relevant information
    Y = data[:, 697]
    X = data[:, 1:697]


    # Use ExtraTreesClassifier
    pipeline = Pipeline([("feature_selection", PCA(50)),
                        ("classify", ExtraTreesClassifier())])
    params = {
            'classify__bootstrap': [False], 
            'classify__class_weight': [None], 
            'classify__criterion': ['gini'], 
            'classify__max_depth': [None],
            'classify__max_features': [None], 
            'classify__max_leaf_nodes': [None], 
            'classify__min_samples_leaf': [3,5,6],
            'classify__min_samples_split': [3], 
            'classify__min_weight_fraction_leaf': [0.08,0.1], 
            'classify__n_estimators': [50,80],
            'classify__n_jobs': [-1], 
            'classify__oob_score': [False], 
            'classify__random_state': [None], 
            'classify__verbose': [0],
            'classify__warm_start': [True]
        }

    # Grid search to find best parameters
    gs1 = GridSearchCV(pipeline, params, verbose=2, refit=True, cv=5, n_jobs = -1)
    gs1.fit(X, Y)
    print "\n\nBest Score: ", gs1.best_score_
    print "\nWith Parameters: ", gs1.best_params_
    pipeline.set_params(**gs1.best_params_)
   
    ## Predict with first classifier
    #y1 = clf.fit(X, y)

    ## See how the classifier scores on training set
    #y_train_pred = [int(y) for y in gs1.predict(X_test)]
    #print ('Accuracy on the training set: {:.2f}'.format(metrics.accuracy_score(y, y_train_pred)))
    #print (metrics.confusion_matrix(y, y_train_pred))
    #print (metrics.classification_report(y, y_train_pred))






    # Load the validation and test data and predict the result
    file_test_data = open("Project3_FeatureEngineering/data/test_validate.csv", "rb")
    test_data = np.loadtxt(file_test_data, delimiter=",")
    ids = test_data[:, 0]
    X_test = test_data[:, 1:697]
    y_test = gs1.predict(X_test)

    # Write output
    pd.DataFrame({'Id': ids.astype(int), 'Label': y_test.astype(int)})\
        .to_csv("Project3_FeatureEngineering/results/out.csv", index=False, columns=['Id', 'Label'])
