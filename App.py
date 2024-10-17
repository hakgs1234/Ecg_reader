import streamlit as st
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load the Hugging Face model
model = ViTForImageClassification.from_pretrained("adilsaleem/ecg-image-multilabel-classification")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Streamlit UI for ECG Upload
st.title("ECG Reader")
st.write("Upload your ECG image and check the heart status.")

uploaded_file = st.file_uploader("Upload an ECG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded ECG Image.", use_column_width=True)
    
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Make predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Decode predictions
    predicted_label = logits.argmax(-1).item()
    st.write(f"Predicted heart status: {'Clear' if predicted_label == 0 else 'Unclear'}")

# Optional: Text input for additional questions
st.text_input("Ask any question about your ECG result", key="user_question")
