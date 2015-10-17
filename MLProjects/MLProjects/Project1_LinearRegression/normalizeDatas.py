import numpy as np


#def normalizeDatas(X, Y):

#    mu_X = np.mean(X,axis=0)
#    sigma_X = np.std(X,axis=0)

#    num_col = X.shape[1]
#    for j in range(num_col):
#        m=np.mean(X[:,j])
#        s=np.std(X[:,j])
#        X[:,j]=(X[:,j]-m)/s

#    mu_Y = np.mean(Y)
#    sigma_Y = np.std(Y)

#    Y =(Y - mu_Y)/sigma_Y

#    return np.array(X), np.array(Y), np.array(mu_X), np.array(sigma_X), mu_Y, sigma_Y
	

#THIS IS YOUR OLD SOLUTION MARTINA, WHICH I ALTERED
#I PUT IT HERE AS COMMENT SINCE I WAS NOT SURE WHETHER 
#YOUR UPDATES ALSO HANDLED THE ISSUES I MENTIONED
def normalizeDatas(X, Y):

    mu_X = np.mean(X,axis=0)
    sigma_X = np.std(X,axis=0)

    for i in range (0, X.shape[1]):
        X[:,i] = X[:,i] - mu_X[0,i]
        X[:,i] = X[:,i] / sigma_X[0,i]

    mu_Y = np.mean(Y,axis=0)
    sigma_Y = np.std(Y,axis=0)
    if sigma_Y == 0:
        Y = 0
    else:
        Y = (Y - mu_Y)/sigma_Y

    return X, Y, np.array(mu_X), np.array(sigma_X), mu_Y, sigma_Y
