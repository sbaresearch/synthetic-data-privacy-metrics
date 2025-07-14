# Example code for uploading BN model to MLFLOW for Synthetic Data Generation

import os
import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel
from DataSynthesizer.DataDescriber import DataDescriber
import pandas as pd
from sys import version_info
import cloudpickle


# Define model class to load the model to MLflow and generate synthetic data (should be implemented by any SDG model)
class EnhancedDataSynthesizerWrapper(mlflow.pyfunc.PythonModel):
    
    # We can add the pre-processing and post-processing in the init method. 
        
    def load_context(self,context):
        
        from DataSynthesizer.DataGenerator import DataGenerator
        
        self.model_uri = context.artifacts["bn_model"]
        self.data_generator = DataGenerator()
        
    
    def predict(self,context,model_input, params=None):
        """
        Generates the synthetic data based on the provided input. 
        """
        
        # Define the number of samples as model input
        number_of_samples = model_input
        
        # Get the path to save synthetic data
        save_path = params.get('save_path','synth.csv')       
        
        # We need to include the code to generate here. 
        self.data_generator.generate_dataset_in_correlated_attribute_mode(number_of_samples,self.model_uri)
        self.data_generator.save_synthetic_data(save_path)
        synthetic_data = pd.read_csv(save_path)
        
        return synthetic_data   


def synthetic_data_model(data_path):
    """
    This function implements the core logic for synthetic data generation. It takes the original dataset as input and returns the local file path of the trained model, which can be used to generate synthetic data. In this implementation, the model is based on a Bayesian Network from the EnhancedDataSynthesizer library, but it should be modified as needed to accommodate other model types.
    """
     # Train model and save it.
    describer_ga = DataDescriber(category_threshold=20)
    describer_ga.describe_dataset_in_correlated_attribute_mode_ga(dataset_file=data_path, k=3)
    model_path = "bn_model.json"
    describer_ga.save_dataset_description_to_file(model_path)
    return model_path


def mlflow_config(model_dependencies, model_local_path, model_name): 
    # Set MLFlow URI
    tracking_uri = os.getenv('MLFLOW_URL')
    print(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    # Get the Python Version
    PYTHON_VERSION=f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    # Create 'artifacts' dictionary, will be passed to mlflow.pyfunc.save_model which will make a copy of the file in the MLflows model's directory.
    artifacts = {"bn_model": model_local_path}

    conda_env = {
        "channels": ["defaults"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": model_dependencies,
            },
        ],
        "name": "sdg_env",
    }

    # Then log the model in a run and register it to MLFlow.
    with mlflow.start_run():
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path=f"{model_name}",
            python_model=EnhancedDataSynthesizerWrapper(),
            artifacts=artifacts,
            conda_env= conda_env, 
            registered_model_name=f"{model_name}"
        )

            



if __name__ == "__main__":

    # Set environment variables ( can be done in console or .venv file and loaded with python-dotenv)
    os.environ["MLFLOW_URL"] = "http://localhost:5000"

    #Load the data 
    data_path = 'notebooks/data/adult_data.csv'
    
    # Train SDG Model locally and return path to trained model
    model_local_path = synthetic_data_model(data_path=data_path)

    # Set all the information to load the model to mlflow 
    model_dependencies = [
                    f"mlflow=={mlflow.__version__}",
                    f"cloudpickle=={cloudpickle.__version__}",
                    "git+https://github.com/sbaresearch/EnhancedDataSynthesizer.git",
                ]

    model_name =  "BN_trained_model"

    mlflow_config(model_dependencies, model_local_path, model_name)



