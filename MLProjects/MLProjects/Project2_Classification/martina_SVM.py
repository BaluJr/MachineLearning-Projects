import numpy as np
import pandas as pd
from sklearn import cross_validation, svm
from sklearn.preprocessing import MinMaxScaler

# Load the training data
file_data = open("/Users/MR/Desktop/ML2/train.csv","rb")
data = np.loadtxt(file_data, delimiter=",")

# Extract ids, target and characteristics
ids = data[:, 0]
y = data[:, 8]
data = data[:, 1:8]

scalerX = MinMaxScaler().fit(data)
X = scalerX.transform(data)

C=1.0
best_classifier, best_score = None, 0
clf_list = [svm.LinearSVC(C=C), svm.SVC(kernel='linear')]
for clf in clf_list:
    clf.fit(X, y)    
    test_error = np.mean(cross_validation.cross_val_score(clf, X, y, cv=6, scoring='f1_weighted'))
    if test_error > best_score:
        best_score, best_classifier = test_error, clf
   
    print 'Test error: ', test_error, '\n\n'
    
print "\n\n###################\nBest classifier is " + str(best_classifier) + " with test error: " + str(best_score) + "\n"
best_classifier.fit(X, y)

# Read test and validation data
file_test_data = open("/Users/MR/Desktop/ML2/validate_and_test.csv","rb")
test_data = np.loadtxt(file_test_data, delimiter=",")
ids = test_data[:, 0]
X_test = test_data[:, 1:8]
X_test = scalerX.transform(X_test)
y_test = best_classifier.predict(X_test)

# Write output file
pd.DataFrame({'Id': ids.astype(int), 'Label': y_test}).to_csv("/Users/MR/Desktop/ML2/out.csv", index=False, columns=['Id', 'Label'])
