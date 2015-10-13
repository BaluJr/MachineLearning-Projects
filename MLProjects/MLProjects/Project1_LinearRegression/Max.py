import numpy as np

# CONFIG ------------------------------------
IGNORED_DIMENSIONS = [] # Input dimensions which shall be ignored
REGULATION_Y = 2 # 0 = Least Squares, 1 = Lasso, 2 = Ridge
REGULATION_LAMBDA = -0.31 # Influence factor of regulation





# DATA HELPERS ------------------------------
def importData (path, validation = False):
    """ Imports the data. 
    Used for trainings and validation datafiles of project1.

    Parameters:
        path - path to the file, that contains the data
        training - defines whether Y is created and returned
        
    Returns:
        vector of IDs
        matrix of Vector for each dim
        vector of Y
        The Vectors are of course as long as the imput file is long

    Raises:
        IOError - when file cannot be found or opened
    """
    ids, targets = [], []
    variables = [[] for x in xrange(14-len(IGNORED_DIMENSIONS))]
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split(",") 
            ids.append(int(line[0]))
            if validation:
                intervall = line[1:]
            else:
                targets.append(float(line[-1]))
                intervall = line[1:-1]

            i = 0
            for j in intervall:
                if j not in IGNORED_DIMENSIONS:
                    variables[i].append(float(j))
                    i+=1                   
    if validation:
        return np.array(ids), np.transpose(np.matrix(variables))
    else:
        return np.array(ids), np.transpose(np.matrix(variables)), np.transpose(np.matrix(targets))


def writeResult (ids, delays, filename):
    """ Write the result to file. 
    Writes the result to the specified file,
    in the kaddle-compatible way.
    
    Parameters:
        ids - array of the ids
        delays - array of the predicted delays
                 in same order as ids
        filename - the filename including the path
    """

    if not filename.endswith('.csv'):
        filename += ".csv"  
    file = open(filename, 'w+')
    file.write("Id,Delay\n")
    for cur in zip(ids, delays):
        file.write(str(cur[0]) + "," + str(cur[1].item(0)) + "\n")



# ESTIMATORS---------------------------------
def leastSquares (X, Y):
    W = np.transpose(X) * X
    W = np.linalg.inv(W)
    W = W * np.transpose(X)
    return W * Y

def ridgeRegression (X, Y):
    W = np.transpose(X) * X + REGULATION_LAMBDA * np.identity(X.shape[1])
    W = np.linalg.inv(W)
    W = W * np.transpose(X)
    return W * Y 



# MAIN PROGRAM ------------------------------
if __name__ == "__main__":
    # Import data
    ids, X, Y = importData("Project1_LinearRegression/data/train.csv");

    # Determin the minimal Beta Vector by LeastSquaresEstimate
    if REGULATION_Y == 0 :
        B = leastSquares(X, Y)
    elif REGULATION_Y == 2 :
        B = ridgeRegression(X, Y)

    # Predict the result
    idsPredict, XPredict = importData("Project1_LinearRegression/data/validate_and_test.csv", validation = True)
    YPredict = XPredict * B

    # Output the result
    writeResult(idsPredict, YPredict, "Project1_LinearRegression/results/max.csv");
