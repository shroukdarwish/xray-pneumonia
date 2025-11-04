import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

st.title("🩻 X-ray Pneumonia Detector")
st.write("Upload a chest X-ray image and I will predict if it shows pneumonia or not.")

# Load model from Google Drive
@st.cache_resource
def load_model():
    url = "https://drive.google.com/uc?id=1rYMbGaJJLt1Zu-rNDnwr0eCOH3N46XbJ"
    output_path = "xray_model.h5"

    if not os.path.exists(output_path):
        with open(output_path, "wb") as f:
            f.write(requests.get(url).content)

    model = tf.keras.models.load_model(output_path)
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("📸 Upload your X-ray image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    st.write("🔍 Analyzing the image...")
    img = preprocess_image(image)

    prediction = model.predict(img)
    result = "🌡️ Pneumonia detected" if prediction[0][0] > 0.5 else "✅ Normal"

    st.subheader("Result:")
    st.success(result)
