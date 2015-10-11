import numpy as np

def importData (path):
    """ Imports the data. 
    Used for trainings and validation datafiles.

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


def writeResult (resultData, path):
    """ Write the result to file. 
    Writes the result to the specified folder,
    in the requested format.
    The file is always called "result.csv",
    

    Parameters:
        resultData - the data, that has to be saved
        path -  path to the folder, where result-
                file shall be located
    """
        table = array2table(resultData, "VariableNames", {"Id", "Delay"})    file = open(path + 'result.csv', 'w+')    writetable(table, path + "result.csv")