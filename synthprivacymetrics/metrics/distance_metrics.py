import numpy as np
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessing import preprocessing_num_cat_features
from sklearn.metrics.pairwise import cosine_similarity

def k_nearest_neighbor(X,Y,k,metric='euclidean'):
    """ Calculates the distances to the k nearest neighbors.

    Parameters
    ----------
    X : pandas.DataFrame
        Training data.
    Y : pandas.DataFrame 
        Query data.
    k : int
        Number of neighbors.
    metric  : (str, optional)
        Metric to compute the nearest_neighbors distance, by default 'euclidean'.

    Returns
    -------
    distances: np.ndarray
        Distances to the k neighbors of each query record.
            
    """
    neighbors=NearestNeighbors(
        metric=metric,
        n_neighbors=k
    ).fit(X)
    distances, _ = neighbors.kneighbors(Y)
    return distances


def distance_to_closest_record(real_data, synth_data, categorical_columns, metric='euclidean',decimals=3):
    """ Measures the distances to the closest real record from each synthetic record and calculates the minimum, mean and standard deviation of distances.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List 
        List with categorical features.
    decimals: (int, optional)
        Number of decimals to round the result.
        
    Returns
    -------
    min_dcr : float
        Distance to the closest record of all synthetic records.
    mean_dcr : float
        Mean of distances to the closest real record from all synthetic records.
    std_dcr : float
        Standard deviation of distances to the closest record of all synthetic records.
    """
    real_data,synth_data = preprocessing_num_cat_features(real_data,synth_data,categorical_columns)
    distances = k_nearest_neighbor(real_data,synth_data, k=1, metric='euclidean')
    min_dcr = round(distances.min(),decimals) 
    mean_dcr = round(distances.mean(),decimals)
    std_dcr = round(distances.std(),decimals)
    DCRResult = namedtuple(
        "DCRResult",
        ("min_dcr","mean_dcr", "std_dcr"),
    )
    return DCRResult(min_dcr,mean_dcr,std_dcr)

def nearest_neighbor_distance_ratio(real_data, synth_data, categorical_columns, metric='euclidean',decimals=3):
    """  Calculates the nearest neighbor distance ratio for each synthetic record by measuring the ratio of the Euclidean distances between a synthetic record 
    and its closest and second-closest neighbors in the real data. The function then computes the minimum, mean, and standard deviation of these ratios.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
    decimals: (int, optional)
        Number of decimals to round the result.
        
    Returns
    -------
    min_nndr : float
        Minimum nearest neighbor distance ratio from synthetic to real records.
    mean_nndr : float
        Mean nearest neighbor distance ratio from synthetic to real records.
    std_nndr : float
        Standard nearest neighbor distance ratio from synthetic to real records.
    """
    real_data,synth_data = preprocessing_num_cat_features(real_data,synth_data,categorical_columns)
    distances = k_nearest_neighbor(real_data,synth_data, k=2, metric='euclidean')
    nndr=np.apply_along_axis(lambda x: x[0]/x[1],1,distances)
    min_nndr = round(nndr.min(),decimals) 
    mean_nndr = round(nndr.mean(),decimals)
    std_nndr = round(nndr.std(),decimals)
    NNDR = namedtuple(
        "NNDRResult",
        ("min_nndr","mean_nndr", "std_nndr"),
    )
    return NNDR(min_nndr,mean_nndr,std_nndr)


def cosine_similarity_metric(real_data,synth_data,categorical_columns,decimals=3):
    """  Calculates the pairwise cosine similarity between real and synthetic records. 
    The function then computes the mean, standard deviation and the maximum value.

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
    decimals: (int, optional)
        Number of decimals to round the result.
        
    Returns
    -------
    mean_cs : float
        Mean cosine similarity between synthetic and real records.
    std_cs : float
        Std cosine similarity between synthetic and real records.
    max_cs : float
        Maximum cosine similarity between a synthetic and real record.
            
    """
    real_data,synth_data = preprocessing_num_cat_features(real_data,synth_data,categorical_columns)
    cos_similarity = cosine_similarity(synth_data,real_data)
    mean_cs = round(cos_similarity.mean(),decimals)
    std_cs = round(cos_similarity.std(),decimals)
    max_cs = round(cos_similarity.max(),decimals)
    CS = namedtuple(
        "CSResult",
        ("mean_cs", "std_cs","max_cs"),
    )
    return CS(mean_cs,std_cs,max_cs)




