import numpy as np


def importData (path):
    """ Imports the data. 
    Used for trainings and validation datafiles of project1.

    Parameters:
        path - path to the file, that contains the data
        
    Returns:
        List of one Numpy-Array per line.
        One entry per data entry within the inner array.

    Raises:
        IOError - when file cannot be found or opened
    """
    result = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            result.append(np.array(line.split(",")));
    return result



def writeResult (resultData, filename):
    """ Write the result to file. 
    Writes the result to the specified file,
    in the kaddle-compatible way.
    
    Parameters:
        resultData - the data, that has to be saved
        filename - the filename including the path
    """
    #if not filename.endswith('.csv'):
    #    filename += ".csv"
    
    #table = array2table(resultData, "VariableNames", {"Id", "Delay"})
    #file = open(path, 'w+')
    #writetable(table, path + "result.csv")
