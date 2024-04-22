import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def preprocessing_num_cat_features(real_data,synth_data,categorical_columns):
    """ Preprocessing pipeline tha scales numerical columns 
    and permorms one-hot encoding of categorical columns in both real and synthetic data. 

    Parameters
    ----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    categorical_columns : List
        List with categorical features.

    Returns
    -------
    real, synth : pandas.DataFrame
        Real and synthetic dataframe encoded.  
    
    """

    # Combined real and synthetic to fit transformer (re-check)
    df =  pd.concat([real_data, synth_data], axis=0)    

    numerical_columns = [ col for col in df.columns if col not in categorical_columns]
    
    # Scale Numerical Features
    num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
    ])

    # Preprocess categorical columns
    cat_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='if_binary'))
    ])

    # Build transformer with preprocessing pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_transformer, numerical_columns),
            ("categorical", cat_transformer, categorical_columns),
        ]
    )   

    preprocessor.fit(df)
    columns_preprocessor=preprocessor.get_feature_names_out()
    
    real_data_preprocess=pd.DataFrame(preprocessor.transform(real_data),columns=columns_preprocessor)
    synth_data_preprocess=pd.DataFrame(preprocessor.transform(synth_data),columns=columns_preprocessor)
   
    return real_data_preprocess,synth_data_preprocess
    

    