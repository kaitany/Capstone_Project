import gradio as gr
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier


#KEY LISTS
expected_inputs = ['TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
       'REGULARITY', 'FREQ_TOP_PACK']

categorical_cols = ['TENURE']

numerical_cols = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
       'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']

#setup

#variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "model\ml.pkl")
#Function to load dataset")

# Load your image file
image_path = os.path.join(DIRPATH, r"images\Understanding Customer Churn.png")  
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

print('preprocessor:', preprocessor)

#Import the model
model = GradientBoostingClassifier()

model = ml_components_dict["model"]


#Function to process inputs and return prediction
def return_prediction(app_image, *args, preprocessor = preprocessor, model = model):

# convert inputs into a dataframe
    #Change the matrix to dataframe
    input_datadf = pd.DataFrame([args], columns=expected_inputs)

    # Preprocess input data
    transformed_input_data = preprocessor.transform(input_datadf)

    # Create DataFrame from the transformed data (dense matrix)
    input_data = pd.DataFrame(transformed_input_data)

    # Define the column names
    numerical_features = ['MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
       'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'REGULARITY', 'FREQ_TOP_PACK']
    
    categorical_features = ['TENURE']

    # Dataframe has no column names after transformation with preprocessor
    # Add back the column names to the columns of the DataFrame
    if hasattr(preprocessor.named_transformers_['cat']['ordinal'], 'get_feature_names_out'):
            
            # Get the ordinal encoded column names
            cat_columns = preprocessor.named_transformers_['cat']['ordinal'].get_feature_names_out(categorical_features)
           
            #Combine the categorical and numerical column names
            all_columns = list(cat_columns) + numerical_features 
            input_data.columns = all_columns

    # Make the prediction using the pipeline containing numerical and categorical transformers
    model_output = model.predict(input_data)

    print('model_output:',model_output)

    if model_output == 1:
        prediction = 1
    else:
        prediction = 0

    # Return the prediction
    return {
        "Prediction: Customer is likely to LEAVE": prediction,
        "Prediction: Customer is likely to STAY": 1 - prediction
    }
  
#Set app interface

#inputs
TENURE = gr.Dropdown(label = "What is the customer's duration in the network?", choices= ['K > 24 month', 'E 6-9 month', 'H 15-18 month', 'G 12-15 month', 'I 18-21 month', 'J 21-24 month', 'F 9-12 month', 'D 3-6 month'])
MONTANT = gr.Slider(label = "What is the customers top-up amount?", minimum=15, maximum=500000, value=20, interactive=True)
FREQUENCE_RECH = gr.Slider(label = "What is the number of times the customer refilled?", minimum=10, maximum=150, value=5, interactive=True)
REVENUE = gr.Slider(label = "What is the customers monthly income?",minimum = 10, maximum=535000, value =5, interactive=True)
ARPU_SEGMENT = gr.Slider(label = "What is the customers income over 90 days/3?",minimum = 10, maximum=200000, value =800, interactive=True)
FREQUENCE = gr.Slider(label = "What is the number of times the customer has made an income?",minimum = 3, maximum=120, value =1, interactive=True)
DATA_VOLUME = gr.Slider(label = "What is the customers number of connections?",minimum = 700, maximum=1900000, value =300, interactive=True)
ON_NET = gr.Slider(label = "What is the customers number of inter Expresso network calls?",minimum = 100, maximum=55000, value =300, interactive=True)
ORANGE = gr.Slider(label = "What is the customers number of calls to Orange network?",minimum = 20, maximum=13000, value =60, interactive=True)
TIGO = gr.Slider(label = "What is the customers number of calls to Tigo network?",minimum = 30, maximum=4300, value =50, interactive=True)
REGULARITY = gr.Slider(label = "What is the number of times the customer is active for 90 days?", minimum = 10, maximum=100, value =10, interactive=True)
FREQ_TOP_PACK = gr.Slider(label = "What is the number of times the customer has activated the top pack packages?", minimum = 7, maximum=700, value =10, interactive=True)


outputs = gr.Label("Awaiting Submission...")
gr.Interface(inputs=[app_image, TENURE, MONTANT, FREQUENCE_RECH, REVENUE,
       ARPU_SEGMENT, FREQUENCE, DATA_VOLUME, ON_NET, ORANGE, TIGO,
       REGULARITY, FREQ_TOP_PACK],
                          fn=return_prediction,
                          outputs=outputs,
                          title="Telecom Churn Prediction App",
                          description="<b>Customer Attrition Prediction App. Input Customer data to get predictions. </b>", live=True).launch(inbrowser=True, show_error=True, share=True)