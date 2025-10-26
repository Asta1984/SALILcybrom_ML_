#Gradio is an open-source Python package that allows you to quickly build a 
# demo or web application for your machine learning model, API, 
# or any arbitrary Python function. You can then share a link to 
# your demo or web application in just a few seconds using Gradio's
#  built-in sharing features. No JavaScript, CSS, or web hosting experience needed!

import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(fn = greet, inputs = ["text", "slider"], outputs=["text"],)

demo.launch(share=True)

