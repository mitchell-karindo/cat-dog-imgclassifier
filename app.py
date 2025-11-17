import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🐱🐶 Cat vs Dog Image Classifier (TFLite Version)")

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)  # (1, 128, 128, 3)
    return image

# Prediction function
def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0][0]  # Sigmoid output

uploaded = st.file_uploader("Upload an image of a **cat or dog**", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("Processing...")

    img_array = preprocess_image(img)
    prediction = predict(img_array)

    if prediction > 0.5:
        st.success(f"🐶 **Dog** ({prediction:.4f})")
    else:
        st.success(f"🐱 **Cat** ({1 - prediction:.4f})")
