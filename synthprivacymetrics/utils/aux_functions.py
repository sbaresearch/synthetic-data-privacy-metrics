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