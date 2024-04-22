import itertools
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from utils.aux_functions import majority_voting


def attribute_disclosure_as_classification(real_data,synth_data, quasi_identifiers, key_length, target, categorical_columns, preprocessor = None, estimators=None, decimals=3):
    """
    Produces results for attribute disclosure attacks based on quasi-identifiers key_length, computes all possible combinations and returns the mean and std accuracy of several classifiers. 
    
    Parameters
    -----------
    real_data : pandas.DataFrame
        Dataframe containing the real data.
    synth_data : pandas.DataFrame
        Dataframe containing the synthetic data.
    quasi_identifiers : list
        List of attributes (possibly) known by the attacker.       
    key_length: int 
        Number of attributes known by the attacker.
    target: str
        Sensitive attribute the attacker tries to infer. 
    categorical_columns: list
        List of categorical attributes in the real and synthetic data.
    preprocessor: (ColumnTransformer,optional)
        ColumnTransformer from sklearn with specific steps to preprocess the data. 
    estimators: (Dict,optional)
        Dict of classifiers implementing fit and predict method as in Sklearn, keys should be the name of the classifiers.
    
    Returns
    -----------
    results_acc: pd.DataFrame
        Dataframe with mean and std accuracy scores from the predictions obtained for the target value given the combinations of quasi-identifiers.
        
     results_f1 : pd.DataFrame
        Dataframe with mean and std f1 scores from the predictions obtained for the target value given the combinations of quasi-identifiers.
        
    """
    
    key_list = list(itertools.combinations(quasi_identifiers, key_length))
    
    # The original dataset is used as reference (Upper Bound)
    data_evaluated = {
        'Original': real_data,
        'Synthetic':synth_data
    }
    
    # Define classifiers used in the task
    if estimators == None:
        estimators = {
            'RF': RandomForestClassifier(),
            'SVM': SVC(),
            'NB':  GaussianNB(),
            'KNN': KNeighborsClassifier(),
            'LR':  LogisticRegression()
        }
    
    # The dummy classifier is used as reference (Lower Bound)
    baseline =  DummyClassifier()
    
    # Preprocessing step in case is not defined. 
    if preprocessor == None:
        
        numerical_columns= [ col for col in real_data.columns if col not in categorical_columns]
        
        # Preprocess numerical columns
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocess categorical columns
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Build transformer with preprocessing pipelines.
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", num_transformer, numerical_columns),
                ("categorical", cat_transformer, categorical_columns),
            ]
        )   
    
    results_acc = pd.DataFrame(index=data_evaluated.keys())
    results_f1 = pd.DataFrame(index=data_evaluated.keys())
            
    for data_name,dataset in data_evaluated.items():
    
        train_data=pd.DataFrame(preprocessor.fit_transform(dataset),columns=numerical_columns+categorical_columns)
        test_data=pd.DataFrame(preprocessor.transform(real_data),columns=numerical_columns+categorical_columns)
        
        # Keys combinations    
        for i,key in enumerate(key_list):
    
            X_train = train_data[list(key)]
            X_test = test_data[list(key)]
            
            y_train = train_data.loc[:,target].values          
            y_test = test_data.loc[:,target].values
            
            # Classifiers training 
            y_pred_all = []
            for model_name,model in estimators.items():
                model.fit(X_train,y_train)
                y_pred=model.predict(X_test)
                y_pred_all.append(y_pred)
                accuracy= accuracy_score(y_test,y_pred)
                f1_val = f1_score(y_test,y_pred)
                results_acc.at[data_name,f'key{i}-{model_name}'] = accuracy
                results_f1.at[data_name,f'key{i}-{model_name}'] = f1_val
            
            # Ensemble results
            y_pred=majority_voting(y_pred_all)
            accuracy= accuracy_score(y_test,y_pred)
            f1_val= f1_score(y_test,y_pred)
            results_acc.at[data_name,f'key{i}-ENS'] = accuracy
            results_f1.at[data_name,f'key{i}-ENS'] = f1_val
                
        # Baseline
        baseline.fit(X_train,y_train)
        y_pred=baseline.predict(X_test)
        accuracy= accuracy_score(y_test,y_pred)
        f1_val= f1_score(y_test,y_pred)
        results_acc.at[data_name,'Dummy']=accuracy
        results_f1.at[data_name,'Dummy']=f1_val
        
    # Average results
    for model_name,model in estimators.items():
        results_acc[f'{model_name}_mean']=results_acc.filter(regex=model_name).mean(axis=1).round(decimals)
        results_acc[f'{model_name}_std']=results_acc.filter(regex=model_name).std(axis=1).round(decimals)
        results_f1[f'{model_name}_mean']=results_f1.filter(regex=model_name).mean(axis=1).round(decimals)
        results_f1[f'{model_name}_std']=results_f1.filter(regex=model_name).std(axis=1).round(decimals)
        
    results_acc[f'ENS_mean']=results_acc.filter(regex='ENS').mean(axis=1).round(decimals)
    results_acc[f'ENS_std']=results_acc.filter(regex='ENS').std(axis=1).round(decimals)
    
    results_f1[f'ENS_mean']=results_f1.filter(regex='ENS').mean(axis=1).round(decimals)
    results_f1[f'ENS_std']=results_f1.filter(regex='ENS').std(axis=1).round(decimals)

    results_acc=results_acc[results_acc.columns.drop(list(results_acc.filter(regex='key')))]
    results_f1=results_f1[results_f1.columns.drop(list(results_f1.filter(regex='key')))]
            
    return results_acc, results_f1
            
