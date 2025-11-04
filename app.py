import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page title
st.title("🩻 X-ray Pneumonia Detector")
st.write("Upload a chest X-ray image and the model will predict whether it shows pneumonia or not.")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("xray_model.h5")
    return model

model = load_model()

# Preprocess the image before prediction
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the same size used in training
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# File uploader
uploaded_file = st.file_uploader("📸 Upload your X-ray image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    st.write("🔍 Analyzing the image...")
    img = preprocess_image(image)
    
    prediction = model.predict(img)
    result = "🌡️ Pneumonia Detected" if prediction[0][0] > 0.5 else "✅ Normal"
    
    st.subheader("Result:")
    st.success(result)
