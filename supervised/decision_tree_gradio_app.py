import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os 

# --- 1. Load the Model ---
# Note: Ensure 'decision_tree.pkl' is in the same directory as this script.
try:
    model = joblib.load('decision_tree.pkl')
    # The model predicts the LabelEncoded 'Risk', where 0 is typically 'bad' and 1 is 'good'.
except FileNotFoundError:
    # This block is for robustness, assuming the user already has the file.
    # If the model file is not found, the app cannot run.
    print("Error: The model file 'decision_tree.pkl' was not found.")
    print("Please ensure the model training script is run first.")
    model = None


# --- 2. Define Feature Names and Prediction Logic ---

# UPDATED: This order now correctly reflects the columns used during model training:
# ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 
#  'Credit amount', 'Duration', 'Purpose'] as determined by X.columns.
MODEL_FEATURE_ORDER = [
    'Age', 
    'Sex', 
    'Job', 
    'Housing', 
    'Saving accounts', 
    'Checking account', 
    'Credit amount', 
    'Duration', 
    'Purpose'
]

def predict(age, sex, job, housing, saving_accounts, checking_account, credit_amount, duration, purpose):
    """
    Predicts the credit risk (Good/Bad) based on the input features.
    
    The inputs are passed directly in the order defined by the Gradio interface, 
    which now matches the model's expected order (MODEL_FEATURE_ORDER).
    
    Returns: A list of tuples [(text, entity_key)] for gr.HighlightedText.
    """
    if model is None:
        # Return error message in the format expected by HighlightedText
        return [("Model not loaded. Check server logs.", "High Risk (Bad Credit)")]

    # 1. Collect inputs in the correct order (Gradio inputs already match MODEL_FEATURE_ORDER)
    ordered_features = [
        age,                # Index 0: 'Age'
        sex,                # Index 1: 'Sex'
        job,                # Index 2: 'Job'
        housing,            # Index 3: 'Housing'
        saving_accounts,    # Index 4: 'Saving accounts'
        checking_account,   # Index 5: 'Checking account'
        credit_amount,      # Index 6: 'Credit amount'
        duration,           # Index 7: 'Duration'
        purpose,            # Index 8: 'Purpose'
    ]

    # 2. Create a Pandas DataFrame to maintain feature names (Best Practice for sklearn models)
    input_data = pd.DataFrame([ordered_features], columns=MODEL_FEATURE_ORDER)
    
    try:
        # 3. Make the prediction (returns 0 or 1)
        prediction_label_encoded = model.predict(input_data)[0]
        
        # 4. Map the label back to a meaningful result and define the entity key
        if prediction_label_encoded == 1:
            result_text = "Low Risk (Good Credit)"
            entity_key = "Low Risk (Good Credit)" # Key matches the color_map entry
        else:
            result_text = "High Risk (Bad Credit)"
            entity_key = "High Risk (Bad Credit)" # Key matches the color_map entry
            
        # 5. Return the result in the format expected by gr.HighlightedText: [(text, entity_key)]
        return [(result_text, entity_key)]
        
    except Exception as e:
        # Handle error by returning a single error message tuple
        return [("Prediction failed due to exception: " + str(e), "High Risk (Bad Credit)")]


# --- 3. Gradio Interface Definition ---

# Define the inputs, ensuring the labels and types guide the user on the expected numerical inputs 
# (which represent the encoded categorical features).
inputs = [
    # Numerical features
    gr.Number(label="Age (in years)", value=30, minimum=19, maximum=75),
    gr.Number(label="Sex (0 or 1 for encoded female/male)", value=1, minimum=0, maximum=1),
    gr.Number(label="Job (Encoded: 0-3)", value=2, minimum=0, maximum=3),
    gr.Number(label="Housing (Encoded: 0-2)", value=1, minimum=0, maximum=2),
    
    # Encoded categorical features
    gr.Number(label="Saving accounts (Encoded: 0-4)", value=2, minimum=0, maximum=4),
    gr.Number(label="Checking account (Encoded: 0-2)", value=1, minimum=0, maximum=2),
    
    # Numerical features
    gr.Number(label="Credit amount (€)", value=1500, minimum=0, maximum=20000),
    gr.Number(label="Duration (in months)", value=12, minimum=4, maximum=72),
    gr.Number(label="Purpose (Encoded: 0-7)", value=4, minimum=0, maximum=7),
]

# Define the HighlightedText component once with the color map.
# The function 'predict' will now return the required data structure: [(text, entity_key)].
formatted_output = gr.HighlightedText(
    label="Credit Risk Assessment",
    color_map={
        "Low Risk (Good Credit)": "rgba(4, 120, 87, 0.8)",  # Green
        "High Risk (Bad Credit)": "rgba(185, 28, 28, 0.8)"   # Red
    }
)

# Create the interface
if model:
    demo = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=formatted_output,
        title="German Credit Risk Decision Tree Predictor",
        description="""Enter the numerical and encoded features below to predict the credit risk.
        Note: The categorical features (Sex, Job, Housing, Saving/Checking accounts, Purpose) 
        must be entered using their numerical encoded values (0, 1, 2, etc.) as they were during training.
        """,
    )

    demo.launch()
else:
    print("Gradio launch skipped because the model could not be loaded.")

    