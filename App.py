import streamlit as st
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import tensorflow as tf

# Load the Hugging Face model and image processor
model_name = "adilsaleem/ecg-image-multilabel-classification"  # Verify this model name
processor = ViTImageProcessor.from_pretrained(model_name)
model = tf.keras.models.load_model('path/to/your/model')  # Ensure you have the TensorFlow model

def preprocess_image(image):
    # Resize and preprocess the image for the Hugging Face model
    image = image.resize((224, 224))  # Adjust size based on your model's input
    image_array = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def classify_ecg(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Process the image using the Hugging Face processor
    inputs = processor(images=image, return_tensors="tf")  # Prepare inputs for Hugging Face model
    
    # Perform inference
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest score
    
    return "Clear" if predicted_class[0] == 1 else "Unclear"  # Adjust based on your model's labels

# Streamlit UI for ECG Upload
st.title("ECG Analysis Tool")
st.write("Upload your ECG image and check the heart status.")

uploaded_file = st.file_uploader("Upload an ECG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Image.", use_column_width=True)

    # Make predictions using the Hugging Face model
    heart_status = classify_ecg(image)
    st.write(f"Predicted heart status: {heart_status}")

# Optional: Text input for additional questions
user_question = st.text_input("Ask any question about your ECG result", key="user_question")
if user_question:
    st.write("You asked:", user_question)
    # Placeholder for answering questions
    st.write("This feature is under development.")
