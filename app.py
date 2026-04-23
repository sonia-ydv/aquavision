import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import datetime
import random

st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI")
st.markdown("""
AI-powered tool to:
- Enhance underwater images  
- Detect marine patterns  
- Generate captions  
- Estimate pollution levels  
""")

# -------- IMAGE ENHANCEMENT (No CV2) --------
def enhance_image(image):
    image = ImageEnhance.Color(image).enhance(1.8)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = image.filter(ImageFilter.SHARPEN)
    return image

# -------- FAKE AI DETECTION --------
def detect_objects(image):
    objects = ["Fish", "Coral", "Plastic", "Rock", "Seaweed"]
    detections = []

    for _ in range(random.randint(1, 4)):
        obj = random.choice(objects)
        conf = round(random.uniform(0.5, 0.95), 2)
        detections.append((obj, conf))

    return detections

# -------- SMART CAPTION --------
def generate_caption(detections):
    if not detections:
        return "No visible marine life detected."

    names = [d[0] for d in detections]
    return f"Underwater scene showing {', '.join(names)}."

# -------- POLLUTION SCORE --------
def pollution_index(detections):
    return sum(10 for d in detections if "Plastic" in d[0])

# -------- AGENT DECISION --------
def agent_decision(detections):
    if not detections:
        return "No Objects Detected"

    avg_conf = sum(d[1] for d in detections) / len(detections)

    if avg_conf < 0.6:
        return "⚠️ Low Confidence"
    else:
        return "✅ High Confidence"


# -------- UI --------
uploaded_file = st.file_uploader("📤 Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    enhanced = enhance_image(image)

    with col2:
        st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    with st.spinner("🤖 AI analyzing..."):
        detections = detect_objects(enhanced)
        caption = generate_caption(detections)
        pollution = pollution_index(detections)
        decision = agent_decision(detections)

    st.subheader("📊 Results")

    st.write("**📝 Caption:**", caption)

    st.write("**🎯 Detected Objects:**")
    for obj, conf in detections:
        st.write(f"- {obj} ({conf})")

    st.write(f"**🌍 Pollution Index:** {pollution}")
    st.write(f"**🤖 AI Decision:** {decision}")

    df = pd.DataFrame(detections, columns=["Object", "Confidence"])
    df["Pollution Index"] = pollution
    df["Timestamp"] = datetime.datetime.now()

    st.download_button(
        "📥 Download Report",
        df.to_csv(index=False),
        file_name="report.csv"
    )