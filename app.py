import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

from tensorflow_lite_support.python.task.core import BaseOptions
from tensorflow_lite_support.python.task.vision import ImageClassifier
from tensorflow_lite_support.python.task.processor import ClassificationOptions
from tensorflow_lite_support.python.task.vision import ImageClassifier

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload gambar dan akan memprediksi apakah itu **kucing atau anjing**.")

# ===============================
# LOAD MODEL TFLITE
# ===============================

@st.cache_resource
def load_classifier():
    model_path = "model.tflite"  # pastikan nama file sama di GitHub

    options = BaseOptions(file_name=model_path)
    classifier = ImageClassifier.create_from_options(
        ImageClassifier.ImageClassifierOptions(
            base_options=options,
            classification_options=ClassificationOptions(max_results=2)
        )
    )
    return classifier

classifier = load_classifier()

# ===============================
# IMAGE PREDICT FUNCTION
# ===============================
def predict(img: Image.Image):
    # Simpan sementara agar bisa dibaca oleh TensorImage
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        img.save(temp.name)
        temp_path = temp.name

    image_tensor = ImageClassifier.TensorImage.create_from_file(temp_path)
    result = classifier.classify(image_tensor)

    os.remove(temp_path)

    top_result = result.classifications[0].categories[0]
    label = top_result.label
    score = top_result.score

    return label, score


# ===============================
# UI
# ===============================

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        label, score = predict(img)

        st.success(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{score:.2f}**")
