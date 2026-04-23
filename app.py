import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import datetime


st.set_page_config(page_title="Deep-Sea Vision AI", layout="wide")

st.title("🌊 Deep-Sea Vision AI")
st.markdown("""
AI-powered tool to:
- Enhance underwater images  
- Detect marine objects  
- Generate captions  
- Estimate pollution levels  
""")


@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return yolo, processor, caption_model

yolo_model, processor, caption_model = load_models()


def enhance_image(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

def detect_objects(img):
    results = yolo_model(img)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_model.names[cls]
            detections.append((label, conf))
    return detections

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def pollution_index(detections):
    plastic_keywords = ["bottle", "plastic", "bag"]
    count = sum(1 for d in detections if any(k in d[0].lower() for k in plastic_keywords))
    return count * 10

def agent_decision(detections):
    if not detections:
        return "No Objects Detected"
    avg_conf = sum([d[1] for d in detections]) / len(detections)
    if avg_conf < 0.4:
        return "⚠️ Recheck Image"
    elif avg_conf < 0.7:
        return "🤔 Low Confidence"
    else:
        return "✅ High Confidence"


uploaded_file = st.file_uploader("📤 Upload underwater image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    enhanced = enhance_image(img_cv)

    with col2:
        st.image(enhanced, caption="Enhanced Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image..."):
        try:
            detections = detect_objects(enhanced)
            caption = generate_caption(image)
            pollution = pollution_index(detections)
            decision = agent_decision(detections)
        except Exception as e:
            st.error("Error during processing. Try another image.")
            st.stop()

 
    st.subheader("📊 Results")

    st.write("**📝 Caption:**", caption)

    st.write("**🎯 Detected Objects:**")
    if detections:
        for obj, conf in detections:
            st.write(f"- {obj} ({conf:.2f})")
    else:
        st.write("No objects detected")

    st.write(f"**🌍 Pollution Index:** {pollution}")
    st.write(f"**🤖 AI Decision:** {decision}")

    df = pd.DataFrame(detections, columns=["Object", "Confidence"])
    df["Pollution Index"] = pollution
    df["Timestamp"] = datetime.datetime.now()

    st.download_button(
        "📥 Download Report (CSV)",
        df.to_csv(index=False),
        file_name="deep_sea_report.csv"
    )