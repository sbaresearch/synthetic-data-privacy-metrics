import io
import os
import sys
import mlflow
from mlflow import MlflowClient
import pandas as pd


class GeneratorClass:
       
    def read_input(self):
        """Reads the input file to generate synthetic data"""
        data = sys.stdin.read()
        csv_file = io.StringIO(data)
        self.input_data = pd.read_csv(csv_file)
        print(self.input_data.head())

    def fit(self,dataset):
        """Processes the input data (mlflow function)"""
        self.trained=True
        self.input_data = dataset
    
    def generate(self, num_samples):
        """Generates synthetic data with the given number of samples"""  
        
        # Access environment variables (mlflow url and model url)
        tracking_uri = os.getenv("MLFLOW_URL")
        if tracking_uri is None:
            raise EnvironmentError("MLFLOW_URL is required but not defined.")   
        
        model_name = os.getenv("MODEL_NAME")
        if model_name is None:
            raise EnvironmentError("MLFLOW_NAME is required but not defined.")   

        model_version = os.getenv("MODEL_VERSION")      
        
        # Set the tracking uri of the ml_flow_server
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)       
        
        # Get the last version of the model in case not provided
        if model_version == None:
            model_info = client.get_registered_model(model_name)
            if len(model_info.latest_versions)>1:
                model_version = model_info.latest_versions[0].version
            else:
                model_version=1
        
        # Load the saved model 
        model_uri = f"models:/{model_name}/{model_version}"
        print(model_uri)
        
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Generate data with the model
        output_data = loaded_model.predict(num_samples,{})
        print(output_data)
                
        
        output = output_data.to_csv(index=False)

        # Write the encoded CSV data to stdout as bytes
        sys.stdout.buffer.write(output.encode('utf-8'))
        

if __name__ == "__main__":

    os.environ["MLFLOW_URL"] = "http://localhost:5000"
    os.environ["MODEL_NAME"] = "BN_trained_model"
    
    num_samples = 2000

    instance = GeneratorClass()
    instance.generate(num_samples)