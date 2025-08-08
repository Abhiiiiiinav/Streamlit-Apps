import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="EfficientNetB0 Classifier", layout="centered")

# Load labels
@st.cache_data
def load_labels():
    with open("imagenet_labels.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Load model from local file
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("efficientnetb0_imagenet.h5")

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

# Predict
def predict(image, model, labels):
    processed = preprocess_image(image)
    predictions = model(processed)
    decoded = tf.keras.applications.efficientnet.decode_predictions(predictions.numpy(), top=3)[0]
    return [(label[1], float(label[2])) for label in decoded]

# UI
st.title("🧠 EfficientNetB0 Image Classifier")
st.write("Upload an image and get top-3 predictions from the EfficientNetB0 model (trained on ImageNet).")

uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Running inference..."):
            model = load_model()
            labels = load_labels()
            results = predict(image, model, labels)

        st.subheader("🔎 Top Predictions:")
        for label, score in results:
            st.write(f"**{label}**: {score * 100:.2f}%")
            st.progress(float(score))
