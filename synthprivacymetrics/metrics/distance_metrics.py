import numpy as np
from collections import namedtuple
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from utils.preprocessing import preprocessing_num_cat_features
from sklearn.metrics.pairwise import cosine_similarity
from sdmetrics.single_table import NewRowSynthesis
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import directed_hausdorff

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

def exact_matches (real_data, synthetic_data, numerical_match_tolerance):
    """Determines whether each row in the synthetic data is new or matches the real data.
    For more information, refer to https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/newrowsynthesis

    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    numerical_match_tolerance : float
        Float indicating how close two numerical values should be to considered them as match.      
        
    Returns
    -----------
    score : int
        Value between 0 and 1, representing the proportion of rows that match a row in real data.
        A value of 0 means all rows are copies and 1 there is no matching rows.
    """
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(synthetic_data)
    
    score = NewRowSynthesis.compute(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata = metadata,
    numerical_match_tolerance=numerical_match_tolerance,
    synthetic_sample_size=None 
    )
    
    return  score


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
    Proposed in https://doi.org/10.1055/s-0042-1760247.

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


def outliers_similarity(real_data,synthetic_data, categorical_columns, method='isolation_forest', **kwargs):
    '''
    Calculates the number of outliers that are identical in synthetic and original training data.
    ** Note that tha parameters for the outlier method should be pass to obtain accurate results.
    
    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.
    method: str 
        Method to identify outliers in the real and synthetic data. Possible values: ['isolation_forst','dbscan','lof']. 
        By default, method = 'isolation_forest'.
        
    Returns
    -------
    exact_matches_count : int
        Number of identical records that are outliers in both real and synthetic data.
    
    '''
    real_data, synthetic_data = preprocessing_num_cat_features(real_data,synthetic_data,categorical_columns)
    
    outlier_methods = {
        'isolation_forest': IsolationForest,
        'dbscan': DBSCAN,
        'lof': LocalOutlierFactor
    }
    
    if method in outlier_methods:
        
        # Identify the outliers in the training data
        clf_real = outlier_methods[method]
        outliers=clf_real(**kwargs).fit_predict(real_data)
        real_data['outliers'] = outliers
        
        # Identify the outliers in the synthetic data
        clf_syn = outlier_methods[method]
        outliers=clf_syn(**kwargs).fit_predict(synthetic_data)
        synthetic_data['outliers'] = outliers
        
        distances = k_nearest_neighbor(real_data[real_data['outliers'] == -1],synthetic_data[synthetic_data['outliers'] == -1],k=1)
        
        # Count the number of outliers that are identical matches in the synthetic data.
        exact_matches_count = np.count_nonzero(distances == 0)
        
    else:
        raise ValueError(f"The outlier method {method} is not supported.")
    
    return exact_matches_count


def hausdorff_distance(real_data,synth_data,categorical_columns,decimals=3):
    '''
    Calculates the proximity of the real and synthetic data as the maximum distance from a point in one set to the nearest point in the other set. Two
    sets are closed if for every point in either set, there exists a point in the other set that is arbitrarily close to it.
    Proposed in https://doi.org/10.1055/s-0042-1760247.
    
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
    haus_distance : int
        Number of identical records that are outliers in both real and synthetic data.    
        
    '''
    real_data,synth_data = preprocessing_num_cat_features(real_data,synth_data,categorical_columns)
    haus_distance= round(max(directed_hausdorff(real_data, synth_data)[0], directed_hausdorff(synth_data, real_data)[0]),decimals)
    return haus_distance