# ML Capstone Speech-to-Text and Classification Emergency Case

# InstaHelp ML Model Deployment

This notebook demonstrates how to load and use the trained machine learning model for classifying emergency cases.

### Purpose:
- Classify emergency cases into `high`, `medium`, or `low` levels based on input text.

### Steps:
1. Install required dependencies.
2. Load the trained model.
3. Use the model to classify example cases.

###  Install Dependencies
!pip install tensorflow numpy

##% Load Model
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model_name.h5')

print("Model loaded successfully!")

# Define Prediction Function




