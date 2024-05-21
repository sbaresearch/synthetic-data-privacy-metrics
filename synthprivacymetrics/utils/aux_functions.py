from sklearn.neighbors import BallTree

def majority_voting(predictions_list):
    """ Performs mayority voting from a vector containing predictions from different classifiers. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different classifiers.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on majority voting.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        most_frequent_val = most_frequent(predictions)
        ensemble_result.append(most_frequent_val)
        
    return ensemble_result

def most_frequent(entry_list):
    """
    Counts more frequent element in a list.
    """
    return max(set(entry_list), key=entry_list.count) 

def average_results(predictions_list):
    """ Performs simple average from a vector containing predictions from different regression models. 
    
    Parameters
    ----------
    predictions_list : list
        List containing prediction vectors of different regression models.
    Returns
    -------
    ensemble_result: list
        List containing predictions based on simple average.
    
    """
    ensemble_result=[]
    for i in range(len(predictions_list[0])):
        predictions=[ val[i] for val in predictions_list]
        avg_val = np.mean(predictions)
        ensemble_result.append(avg_val)
        
    return ensemble_result

def GCAPClassifier(X_train, y_train, X_test, key):
    '''Algorithm proposed in https://doi.org/10.1145/3374664.3375722 '''

    bat = BallTree(X_train, metric='hamming')
    y_pred = []
    for entry in range(len(X_test)):
        pt = X_test.values[entry].reshape(1, -1)
        for i in range(len(key) + 1):
            match_list = bat.query_radius(pt, r=i / len(key))[0]
            if len(match_list) != 0:
                break
            
        maj_list = []
        for i in range(len(match_list)):
            maj_list.append(y_train[match_list[i]])
            
            
        y_pred.append(most_frequent(maj_list))

    return y_pred