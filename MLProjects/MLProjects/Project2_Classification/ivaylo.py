import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

# Load data from the train file
file_data = open("data/train.csv", "rb")
data = np.loadtxt(file_data, delimiter=",")

# Take only the relevant information
y = data[:, 8]
data = data[:, 1:8]

# Normalize the data
X = np.log2(data)

# Use RandomForestClassifier
clf = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_depth=None, max_features=None, max_leaf_nodes=None,
           min_samples_leaf=3, min_samples_split=1,
           min_weight_fraction_leaf=0.01, n_estimators=100, n_jobs=-1,
           oob_score=False, random_state=1, verbose=0, warm_start=False)
clf.fit(X, y)

# See how the classifier scores on training set
y_train_pred = clf.predict(X)
print ('Accuracy on the training set: {:.2f}'.format(metrics.accuracy_score(y, y_train_pred)))
print (metrics.confusion_matrix(y, y_train_pred))
print (metrics.classification_report(y, y_train_pred))

# Load the validation and test data and predict the result
file_test_data = open("data/validate_and_test.csv", "rb")
test_data = np.loadtxt(file_test_data, delimiter=",")
ids = test_data[:, 0]
X_test = test_data[:, 1:8]
X_test = np.log2(X_test)
y_test = clf.predict(X_test)

# Write output
pd.DataFrame({'Id': ids.astype(int), 'Label': y_test.astype(int)})\
    .to_csv("data/out.csv", index=False, columns=['Id', 'Label'])
