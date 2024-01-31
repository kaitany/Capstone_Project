import gradio as gr
import requests
import pandas as pd
import numpy as np
import pickle
import uvicorn
from fastapi import HTTPException
import pickle, os

#KEY LISTS
expected_inputs = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
       'REGULARITY', 'FREQ_TOP_PACK']

categorical_cols = ['REGION', 'TENURE']

numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']

# tenure mapping 'TENURE' category
tenure_mapping = {'D 3-6 month':0,
                  'E 6-9 month':1,
                  'F 9-12 month':2,
                  'G 12-15 month':3,
                  'H 15-18 month':4,
                  'I 18-21 month':5,
                  'J 21-24 month':6,
                  'K > 24 month': 7 
}

# Define categories for 'TENURE' and 'REGION'
tenure_categories = list(tenure_mapping.keys())
region_categories = ['DAKAR', 'SAINT-LOUIS', 'THIES', 'LOUGA', 'MATAM', 'FATICK',
       'KAOLACK', 'DIOURBEL', 'TAMBACOUNDA', 'ZIGUINCHOR', 'KOLDA',
       'KAFFRINE', 'SEDHIOU', 'KEDOUGOU']

#setup

#variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
#ml_core_fp = os.path.join(DIRPATH, "model\ml.pkl")
#ml_core_fp = os.path.join(DIRPATH, "model", "ml.pkl")
ml_core_fp = os.path.join(DIRPATH, r"model\ml.pkl")
#Function to load dataset")

# Load your image file
image_path = os.path.join(DIRPATH, r"images/Understanding Customer Churn.png")  
# Create an Image component
app_image = gr.Image(image_path)

#useful functions
def  load_ml_components(fp):
    "load the ml components to re-use in app"
    with open(fp, 'rb') as file:
        obj = pickle.load(file)
        return obj
    
#Execute and instantiate the toolkit components
ml_components_dict = load_ml_components(fp = ml_core_fp)

preprocessor = ml_components_dict["preprocessor"]

#Import the model

model = ml_components_dict["model"]

    # Initialize result variable
result = ""

# Function to update Gradio interface with prediction
def update_interface(app_image, REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE,
                      ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
                      REGULARITY, FREQ_TOP_PACK):
    
    # Initialize result variable
    result = ""

    # Create a dictionary with input data
    input_data = {

        "REGION": REGION,
        "TENURE": TENURE,
        "MONTANT": MONTANT,
        "FREQUENCE_RECH": FREQUENCE_RECH,
        "REVENUE": REVENUE,
        "ARPU_SEGMENT": ARPU_SEGMENT,
        "FREQUENCE": FREQUENCE,
        "DATA_VOLUME": DATA_VOLUME,
        "ON_NET": ON_NET,
        "ORANGE": ORANGE,
        "TIGO": TIGO,
        "REGULARITY": REGULARITY,
        "FREQ_TOP_PACK": FREQ_TOP_PACK
    }


    try:
        # Send a POST request to the FastAPI endpoint
        response = requests.post("http://127.0.0.1:8000/predict_churn", json=input_data)

        # Print the content of response.text for debugging

        # Check if the request was successful
        if response.status_code == 200:
            # Get the prediction from the response
            prediction = response.json()

            # Retrieve values using .get()
            predicted_value = prediction.get('prediction')
            churn_status = prediction.get('churn_status')
          
         #  Check if the values are not None before using them
            if predicted_value is not None and churn_status is not None:

                result = f"prediction: {predicted_value}\nchurn status: {churn_status}"
            else:
                result = "Error: Unable to fetch prediction from FastAPI"

        else:
            # Handle the case where the request was not successful
            print("Error: Unable to fetch prediction from FastAPI")

    except Exception as e:
        # Handle any exceptions that might occur
        raise HTTPException(status_code=500, detail=str(e)) 

        # Return the result
    return result      

#Set app interface

#inputs
REGION = gr.Dropdown(label="What region is the customer located?", choices= region_categories)
TENURE = gr.Dropdown(label = "What is the customer's duration in the network?", choices= tenure_categories)
MONTANT = gr.Slider(label = "What is the customers top-up amount?", minimum=15, maximum=500000, value=20, interactive=True)
FREQUENCE_RECH = gr.Slider(label = "What is the number of times the customer refilled?", minimum=10, maximum=150, value=5, interactive=True)
REVENUE = gr.Slider(label = "What is the customers monthly income?",minimum = 10, maximum=535000, value =5, interactive=True)
ARPU_SEGMENT = gr.Slider(label = "What is the customers income over 90 days/3?",minimum = 10, maximum=200000, value =800, interactive=True)
FREQUENCE = gr.Slider(label = "What is the number of times the customer has made an income?",minimum = 3, maximum=120, value =1, interactive=True)
DATA_VOLUME = gr.Slider(label = "What is the customers number of connections?",minimum = 700, maximum=1900000, value =300, interactive=True)
ON_NET = gr.Slider(label = "What is the customers number of inter Expresso network calls?",minimum = 100, maximum=55000, value =300, interactive=True)
ORANGE = gr.Slider(label = "What is the customers number of calls to Orange network?",minimum = 20, maximum=13000, value =60, interactive=True)
TIGO = gr.Slider(label = "What is the customers number of calls to Tigo network?",minimum = 30, maximum=4300, value =50, interactive=True)
REGULARITY = gr.Slider(label = "What is the number of times the customer is active for 90 days?", minimum = 7, maximum=100, value =5, step=1, interactive=True)
FREQ_TOP_PACK = gr.Slider(label = "What is the number of times the customer has activated the top pack packages?", minimum = 7, maximum=700, value =10, interactive=True)


outputs = gr.Label("Awaiting Submission...")

gr.Interface(inputs=[app_image, REGION, TENURE, MONTANT, FREQUENCE_RECH, REVENUE,
                      ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
                      REGULARITY, FREQ_TOP_PACK],
             outputs=outputs,
             title="Telecom Churn Prediction App",
             description="<b>Customer Attrition Prediction App. Input Customer data to get predictions. </b>",
             fn=update_interface).launch(inbrowser=True, show_error=True, share=True)
