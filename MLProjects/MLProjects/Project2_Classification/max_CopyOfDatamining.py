#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np 
import time
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

# CONFIG
DIMENSION = 400  # Dimension of the original data.
CLASSES = [+1, -1]   # The classes that we are trying to predict.
D = 10000


#precomputed attributes of the data
LEAST_IMPORTANT_DIMENSIONS = [ 5, 12, 20, 25, 26, 33, 36, 37, 38, 40, 49, 50, 67, 68, 74, 75, 84, 87, 89, 91, 94, 96, 97, 99, 101, 105, 106, 107, 110, 117, 
					  123, 125, 137, 139, 148, 152, 153, 154, 166, 167, 168, 169, 173, 177, 188, 193, 199, 201, 202, 203, 206, 214, 217, 220, 223, 
					  225, 235, 236, 242, 244, 245, 248, 250, 253, 254, 256, 257, 258, 272, 280, 284, 285, 287, 295, 297, 300, 302, 308, 313, 316, 
					  325, 326, 330, 337, 338, 339, 353, 355, 356, 360, 367, 372, 380, 382, 385, 386, 387, 390, 395, 399]
VARIANCE = 200

# The call for the transformation
np.random.seed(42)
omegas = np.random.multivariate_normal(np.zeros(DIMENSION-100),VARIANCE*np.eye(DIMENSION-100),D)
bs = 2. *np.pi*np.random.random(D)
prefactor = np.sqrt(2./1000.) # precalculate for performance issues
def transform(x_original):
	x_transformed = x_original
	x_transformed = np.delete(np.array(x_transformed), LEAST_IMPORTANT_DIMENSIONS)
	return prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_transformed)+b) ,omegas,bs))




if __name__ == "__main__":

	# read and format data
	data = sys.stdin.read()

	# prepare data
	data = np.fromstring(data, sep=' ')
	data = np.reshape(data,(-1,DIMENSION+1))
	Y = data[:, 0]
	X = data[:, 1:DIMENSION+1]

	# Do the transformation for each x
	x_transformed = np.array([transform(x) for x in X])
    #abgabe_params = {loss='modified_huber', 'alpha':[0.00000001], 'fit_intercept':[False], 'average':[True], 'n_iter':[8], 'n_jobs':[-1], 'penalty':["l2"]}
	#clf = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.00000001, fit_intercept=False, average=True, n_iter=8, n_jobs=-1)
    
   	clf1 = SGDClassifier(loss='modified_huber', alpha=1e-08, fit_intercept=False, average=True, n_iter=8, n_jobs=-1)#, learning_rate="constant", eta0=0.1) 	
   	
	clf1.fit(x_transformed, Y)
	result = ' '.join(' '.join(str(cell) for cell in row) for row in clf1.coef_ )
	print result
