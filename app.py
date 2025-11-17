import streamlit as st
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐶", layout="centered")

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cats_dogs_model.keras", compile=False)
    return model

model = load_model()



# Auto detect image size from model
input_shape = model.input_shape  # (None, H, W, C)
IMG_SIZE = (input_shape[1], input_shape[2])   # (H, W)

# ===============================
# Prediction Function
# ===============================
def predict_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "Dog" if prediction >= 0.5 else "Cat"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    return label, float(confidence)

# ===============================
# Streamlit UI
# ===============================
st.title("🐱🐶 Cats vs Dogs Image Classifier")
st.write("Upload gambar dan model akan memprediksi apakah itu **Kucing** atau **Anjing**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(img)
        st.success(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")
