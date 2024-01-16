from fastapi import FastAPI, HTTPException
import uvicorn
#from typing import List, Literal
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle, os

 #setup
# Get the directory of the current file (FastAPI application file)
DIRPATH = os.path.dirname(os.path.realpath(__file__))

# Construct the path to ml.pkl relative to the current file using forward slashes
ml_core_fp = os.path.join(DIRPATH, r"model\ml.pkl")

#useful functions
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
# Loading: Execute and instantiate ml components
ml_components_dict = load_ml_components(fp = ml_core_fp)

preprocessor = ml_components_dict["preprocessor"]

model =  ml_components_dict["model"]


#API
app = FastAPI(
    title="Customer Churn Analysis API")

# Input for Modelling 
class ChurnFeatures(BaseModel):
    REGION: str = Field(..., description="The location of each client")
    TENURE: str = Field(..., description="Customer's duration in the network")
    MONTANT: float = Field(..., description="Customer's top-up amount")
    FREQUENCE_RECH: float = Field(..., description="Number of times the customer refilled")
    REVENUE: float = Field(..., description="Monthly income of each client")
    ARPU_SEGMENT: float = Field(..., description="Customer's income over 90 days/3")
    FREQUENCE: float = Field(..., description="Number of times the client has made an income")
    DATA_VOLUME: float = Field(..., description="Customer's number of connections")
    ON_NET: float = Field(..., description="Inter expresso call")
    ORANGE: float = Field(..., description="Call to Orange")
    TIGO: float = Field(..., description="Call to Tigo")
    REGULARITY: float= Field(..., description="Number of times the client is active for 90 days")
    FREQ_TOP_PACK: float = Field(..., description="Number of times the customer has activated the top pack packages")


@app.get('/')
def home():
   return "Customer churn"


@app.get('/info')
def appinfo():
    return 'Customer churn prediction: This is my interface'


@app.post('/predict_churn')
def predict_churn(churn_features: ChurnFeatures):
    print("Inside predict_churn function")

    try:

        #Dataframe creation
        df = pd.DataFrame([churn_features.model_dump()])

        print(f'churn_features: {churn_features}')
        df = pd.DataFrame([churn_features.model_dump()])
        print(f'dataframe: {df}')

        #preprocess data
        preprocessed_data = preprocessor.transform(df)

        print(f'Preprocessed data: {preprocessed_data}')

        # Create DataFrames of the preprocessed data
        transformed_df = pd.DataFrame(preprocessed_data)

        #Add back column names to the dataframes

        #Define column names
        categorical_cols = ['TENURE', 'REGION']  
        numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']


        # Get the one-hot encoder transformer
        onehot_encoder = preprocessor.named_transformers_['region']['onehot']

        # Get the ordinal encoder transformer
        ordinal_encoder = preprocessor.named_transformers_['tenure']['ordinal']

        # Access feature names for one-hot encoder
        if hasattr(onehot_encoder, 'get_feature_names_out'):
            cat_columns_onehot = onehot_encoder.get_feature_names_out(['REGION'])

        # Access feature names for ordinal encoder
        if hasattr(ordinal_encoder, 'get_feature_names_out'):
            cat_columns_ordinal = ordinal_encoder.get_feature_names_out(['TENURE'])

        # Combine all feature names
        all_feature_names = np.concatenate([cat_columns_onehot, cat_columns_ordinal, numerical_cols])

        transformed_df.columns = all_feature_names

        # ML prediction
        prediction = model.predict(transformed_df)

            # Interpret prediction based on probability score
        churn_status = " Customer will Churn" if prediction == 1 else "Customer will not Churn" 

        # Create a dictionary for the response
        prediction_data = {
        "selected_model": model,
        "prediction": prediction,
        "churn_status": churn_status,
        }

        return prediction_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction {str(e)}")
    
if __name__ =="__main__":
    uvicorn.run("main:app", reload=True)