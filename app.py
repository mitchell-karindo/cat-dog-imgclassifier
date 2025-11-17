import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.title("🐱🐶 Cat vs Dog Classifier")

@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

def preprocess_image(img, target_size=(128, 128)):
    img = img.resize(target_size)
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return float(output[0][0])

uploaded = st.file_uploader("Upload photo:", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img)
    result = predict(img_array)

    if result > 0.5:
        st.success(f"🐶 Dog ({result:.4f})")
    else:
        st.success(f"🐱 Cat ({1 - result:.4f})")
