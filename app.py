import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI (ML Powered)")

# -------------------------
# LOAD MODEL (CACHED)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pollution_model.h5")

model = load_model()

labels = ["Clean Water", "Moderate Pollution", "High Pollution"]

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# -------------------------
# PREDICT
# -------------------------
def predict(img):
    pred = model.predict(preprocess(img))[0]
    idx = np.argmax(pred)
    return labels[idx], float(np.max(pred))

# -------------------------
# INPUT
# -------------------------
file = st.file_uploader("Upload Underwater Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original", use_column_width=True)

    result, conf = predict(img)

    with col2:
        st.image(image, caption=f"Prediction: {result}", use_column_width=True)

    st.subheader("🧠 AI Analysis")
    st.write("Prediction:", result)
    st.write("Confidence:", round(conf * 100, 2), "%")