# import gradio as gr

# #A simple Function 
# def greet(name):
#     return f"Hello, {name} !"

# #create interface
# demo = gr.Interface(fn= greet, inputs=gr.Textbox(label="Enter your name"), outputs = "text", title = "Hello from gradio")

# demo.launch()

##Number numeric input

# import gradio as gr

# def square(x):
#     return x**2

# demo = gr.Interface(fn=square, inputs=gr.Number(label="Enter a number"), 
#     outputs="number")
# demo.launch()



import numpy as np
import gradio as gr
import joblib
import cv2


# Load trained model 
model = joblib.load("Logistic_model_number_det.pkl")


def predict_number(frame):
    if frame is None:
        return frame, "No video feed" 
    
    mirrored_frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to 8x8 (64 features = 8x8 image)
    resized = cv2.resize(gray, (8, 8))
    
    # Normalize and flatten
    normalized = resized / 255.0
    flattened = normalized.reshape(1, -1)
    
    # Predict
    prediction = model.predict(flattened)
    
    return mirrored_frame, f"Predicted Number: {prediction}"



# # Create interface
demo = gr.Interface(
    fn=predict_number,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=[
        gr.Image(label="Mirrored Feed"),
        gr.Textbox(label="Result")
    ],
    live=True
)


demo.launch(share=True)
