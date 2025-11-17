import streamlit as st
st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐶", layout="centered")

import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ===================================================
# GOOGLE DRIVE MODEL DOWNLOAD
# ===================================================

MODEL_URL = "https://drive.google.com/file/d/1AYzSzCl6NIqyHnPEme0s24VsZRimZgS-/view?usp=drive_link"
MODEL_PATH = "cats_dogs_model.keras"


@st.cache_resource
def load_model():
    # Download model jika belum ada
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()


# ===================================================
# Prediction Function
# ===================================================

IMG_SIZE = (150, 150)  # sesuai input shape model kamu

def predict_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    label = "Cat" if prediction < 0.5 else "Dog"
    confidence = prediction if prediction >= 0.5 else 1 - prediction

    return label, float(confidence)


# ===================================================
# STREAMLIT UI
# ===================================================

st.title("🐱🐶 Cats vs Dogs Image Classifier")
st.write("Upload gambar untuk diprediksi apakah itu **Kucing** atau **Anjing**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, confidence = predict_image(img)
        st.success(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")


